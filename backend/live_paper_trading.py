import ccxt
import pandas as pd
import pandas_ta as ta
import numpy as np
import time
from datetime import datetime
import pytz
from sb3_contrib import RecurrentPPO
from envs.trading_env import BitcoinTradingEnv 
import warnings
import sys

# Ignorar avisos
warnings.filterwarnings("ignore")

# --- CONFIGURAÇÕES ---
SYMBOL = 'BTC/USDT'     
TIMEFRAME = '5m'        
MODEL_PATH = "models/sniper_pro_finished"
INITIAL_BALANCE = 1000  
FEE_RATE = 0.0005       

# --- CONEXÃO ---
print("--- INICIANDO SNIPER EM MODO PAPER TRADING (DADOS REAIS) ---")
print("Conectando à Binance Mainnet (API Pública)...")
try:
    exchange = ccxt.binance({
        'enableRateLimit': True, 
        'options': {'defaultType': 'future'} 
    })
    exchange.load_markets() # Teste de conexão
except Exception as e:
    print(f"Erro ao conectar na Binance: {e}")
    sys.exit()

# --- ENGENHARIA DE FEATURES (COM CÁLCULO MANUAL DO BBP) ---
def calculate_features(df):
    try:
        df = df.copy()
        
        # 1. Log Return
        df['log_ret'] = np.log(df['close'] / df['close'].shift(1))
        
        # 2. RSI 14
        df['rsi'] = ta.rsi(df['close'], length=14)
        
        # 3. RSI Slope
        df['rsi_slope'] = df['rsi'].diff()
        
        # 4. MACD
        macd = ta.macd(df['close'])
        # Pega a coluna do histograma dinamicamente (MACDh...)
        macd_col = [c for c in macd.columns if c.startswith('MACDh') or c.startswith('MACDH')][0]
        df['macd_diff'] = macd[macd_col]
        
        # 5. Bollinger Bands (CORREÇÃO MANUAL AQUI)
        bb = ta.bbands(df['close'], length=20, std=2)
        
        # Identifica as colunas que EXISTEM (Upper e Lower)
        upper_col = [c for c in bb.columns if c.startswith('BBU')][0]
        lower_col = [c for c in bb.columns if c.startswith('BBL')][0]
        width_col = [c for c in bb.columns if c.startswith('BBB')][0] # Largura já vem pronta
        
        # CÁLCULO MANUAL DO %B (Percent B)
        # Fórmula: (Preço - Banda Inferior) / (Banda Superior - Banda Inferior)
        df['bb_pband'] = (df['close'] - bb[lower_col]) / (bb[upper_col] - bb[lower_col])
        
        # Largura da Banda
        df['bb_width'] = bb[width_col]
        
        # 6. Distância das EMAs
        df['ema50'] = ta.ema(df['close'], length=50)
        df['ema200'] = ta.ema(df['close'], length=200)
        df['dist_ema50'] = (df['close'] - df['ema50']) / df['ema50']
        df['dist_ema200'] = (df['close'] - df['ema200']) / df['ema200']
        
        # 7. ATR
        df['atr'] = ta.atr(df['high'], df['low'], df['close'], length=14)
        df['atr_pct'] = df['atr'] / df['close']
        
        df = df.dropna()
        
        feature_cols = [
            'log_ret', 'rsi', 'rsi_slope', 'macd_diff', 
            'bb_pband', 'bb_width', 'dist_ema50', 
            'dist_ema200', 'atr_pct'
        ]
        
        return df[feature_cols], df['close']

    except Exception as e:
        # Se der erro no cálculo (ex: dados insuficientes no começo), retorna vazio
        return None, None

# --- CARREGAR MODELO ---
print(f"Carregando Modelo: {MODEL_PATH}")
try:
    dummy_df = pd.DataFrame({'close': [100]*200}) 
    for col in ['log_ret', 'rsi', 'rsi_slope', 'macd_diff', 'bb_pband', 'bb_width', 'dist_ema50', 'dist_ema200', 'atr_pct']:
        dummy_df[col] = 0.0
    
    dummy_env = BitcoinTradingEnv(dummy_df)
    model = RecurrentPPO.load(MODEL_PATH, env=dummy_env)
    print(">>> Cérebro Carregado e Pronto para o Mercado Real.")
except Exception as e:
    print(f"ERRO FATAL AO CARREGAR MODELO: {e}")
    sys.exit()

# --- ESTADO INICIAL ---
balance = INITIAL_BALANCE
position = 0 
entry_price = 0.0
lstm_states = None
episode_starts = np.ones((1,), dtype=bool)
br_tz = pytz.timezone('America/Sao_Paulo')

print(f"\nSaldo Inicial Fictício: ${balance:.2f} USDT")
print("Aguardando fechamento de candles...\n")

# --- LOOP PRINCIPAL ---
while True:
    try:
        # Baixa 300 candles para garantir cálculo correto
        ohlcv = exchange.fetch_ohlcv(SYMBOL, timeframe=TIMEFRAME, limit=300)
        df_raw = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df_raw['timestamp'] = pd.to_datetime(df_raw['timestamp'], unit='ms')
        
        current_price = df_raw.iloc[-1]['close'] 
        
        # Calcula indicadores
        df_features, df_prices = calculate_features(df_raw)
        
        if df_features is None or len(df_features) == 0:
            print(f"\rAguardando dados suficientes... (Baixados: {len(df_raw)})", end="")
            time.sleep(10)
            continue

        obs = df_features.iloc[-1].values
        
        # Previsão
        action, lstm_states = model.predict(
            obs, 
            state=lstm_states, 
            episode_start=episode_starts, 
            deterministic=True
        )
        episode_starts = np.zeros((1,), dtype=bool)
        
        # --- CORREÇÃO DO ERRO LOOP ---
        # Usa .item() que funciona tanto para arrays 0-d quanto números normais
        if isinstance(action, np.ndarray):
            act_idx = action.item()
        else:
            act_idx = action
        # -----------------------------
        
        target_pos = 0
        if act_idx == 1: target_pos = 1   
        elif act_idx == 2: target_pos = -1 
        
        timestamp_br = datetime.now(br_tz).strftime('%H:%M:%S')
        
        # Lógica de Execução
        if target_pos != position:
            pnl = 0
            if position != 0:
                change_pct = (current_price - entry_price) / entry_price
                if position == 1: pnl = balance * change_pct
                elif position == -1: pnl = balance * (-change_pct)
                
                fee = balance * FEE_RATE
                balance += pnl - fee
                print(f"\n[{timestamp_br}] 💰 FECHOU POSIÇÃO | Preço: {current_price:.2f} | PnL: ${pnl:.2f} | Taxa: -${fee:.2f}")

            if target_pos != 0:
                fee = balance * FEE_RATE
                balance -= fee
                entry_price = current_price
                type_str = "LONG 🟢" if target_pos == 1 else "SHORT 🔴"
                print(f"\n[{timestamp_br}] 🚀 ABRIU {type_str} | Preço: {current_price:.2f} | Taxa: -${fee:.2f}")
            else:
                print(f"\n[{timestamp_br}] 💤 ZEROU POSIÇÃO (Neutro)")

            position = target_pos
            print(f"   >>> SALDO ATUALIZADO: ${balance:.2f} USDT")
            print("-" * 50)
            
        else:
            status = "NEUTRO"
            unrealized_pnl = 0
            if position == 1: 
                status = f"LONG (Desde {entry_price:.2f})"
                unrealized_pnl = balance * ((current_price - entry_price) / entry_price)
            elif position == -1: 
                status = f"SHORT (Desde {entry_price:.2f})"
                unrealized_pnl = balance * (-(current_price - entry_price) / entry_price)
            
            color = "\033[92m" if unrealized_pnl >= 0 else "\033[91m"
            reset = "\033[0m"
            
            # Print dinâmico na mesma linha
            print(f"\r[{timestamp_br}] BTC: {current_price:.2f} | Bot: {status} | PnL Aberto: {color}${unrealized_pnl:.2f}{reset}", end="")

        time.sleep(10) # Checa a cada 10 segundos
        
    except Exception as e:
        print(f"\n[ERRO LOOP]: {e}")
        time.sleep(5)