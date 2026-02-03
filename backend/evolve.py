import pandas as pd
import pandas_ta as ta
import numpy as np
import os
from sb3_contrib import RecurrentPPO
from envs.trading_env import BitcoinTradingEnv

# --- CONFIGURAÇÕES ---
OLD_MODEL_PATH = "models/sniper_pro_finished"
NEW_MODEL_PATH = "models/sniper_pro_v2" # Salva uma nova versão
DATA_FILE = "data/live_market_data.csv"

def calculate_features(df):
    df['log_ret'] = np.log(df['close'] / df['close'].shift(1))
    df['rsi'] = ta.rsi(df['close'], length=14)
    df['rsi_slope'] = df['rsi'].diff()
    macd = ta.macd(df['close'])
    macd_col = [c for c in macd.columns if c.startswith('MACDh')][0]
    df['macd_diff'] = macd[macd_col]
    bb = ta.bbands(df['close'], length=20, std=2)
    upper_col = [c for c in bb.columns if c.startswith('BBU')][0]
    lower_col = [c for c in bb.columns if c.startswith('BBL')][0]
    width_col = [c for c in bb.columns if c.startswith('BBB')][0]
    df['bb_pband'] = (df['close'] - bb[lower_col]) / (bb[upper_col] - bb[lower_col])
    df['bb_width'] = bb[width_col]
    df['ema50'] = ta.ema(df['close'], length=50)
    df['ema200'] = ta.ema(df['close'], length=200)
    df['dist_ema50'] = (df['close'] - df['ema50']) / df['ema50']
    df['dist_ema200'] = (df['close'] - df['ema200']) / df['ema200']
    df['atr'] = ta.atr(df['high'], df['low'], df['close'], length=14)
    df['atr_pct'] = df['atr'] / df['close']
    return df.dropna()

def evolve():
    print("--- INICIANDO PROCESSO DE EVOLUÇÃO ---")
    
    if not os.path.exists(DATA_FILE):
        print("❌ Erro: Nenhum dado novo coletado ainda. Deixe o bot rodar por mais tempo!")
        return

    # 1. Carrega dados novos coletados em tempo real
    print("1. Carregando dados vividos pelo bot...")
    df = pd.read_csv(DATA_FILE)
    print(f"   > {len(df)} candles novos encontrados.")
    
    if len(df) < 500:
        print("⚠️ Dados insuficientes para treino seguro. Espere acumular pelo menos 500 candles.")
        return

    # 2. Calcula indicadores
    print("2. Calculando novos indicadores...")
    df_processed = calculate_features(df)
    
    # 3. Carrega o ambiente e o modelo antigo
    print("3. Carregando cérebro antigo...")
    env = BitcoinTradingEnv(df_processed)
    model = RecurrentPPO.load(OLD_MODEL_PATH, env=env)
    
    # 4. Fine-Tuning (Treino Rápido)
    print("4. EVOLUINDO: Aprendendo com os dados novos...")
    # Rodamos poucos timesteps só para ajustar, não para esquecer o passado
    model.learn(total_timesteps=10000) 
    
    # 5. Salva o modelo novo
    print(f"5. Salvando Sniper Pro V2 em {NEW_MODEL_PATH}...")
    model.save(NEW_MODEL_PATH)
    print("✅ EVOLUÇÃO CONCLUÍDA! Agora atualize o server.py para usar o modelo V2.")

if __name__ == "__main__":
    evolve()