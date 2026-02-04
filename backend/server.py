import asyncio
import json
import os
import threading
import time
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import ccxt.async_support as ccxt
import pandas as pd
import pandas_ta as ta
import numpy as np
from datetime import datetime
from sb3_contrib import RecurrentPPO
from envs.trading_env import BitcoinTradingEnv
import warnings
import gc

warnings.filterwarnings("ignore")

app = FastAPI()

@app.get("/")
def read_root():
    return {
        "status": "ONLINE",
        "message": "Sniper Pro Backend está rodando na Nuvem! 🚀",
        "region": "Frankfurt (EU)",
        "time": datetime.now().strftime("%H:%M:%S")
    }


app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
)

def clean_data(obj):
    if isinstance(obj, (np.integer, np.int64, np.int32)): return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)): return float(obj)
    elif isinstance(obj, np.ndarray): return obj.tolist()
    elif isinstance(obj, dict): return {k: clean_data(v) for k, v in obj.items()}
    elif isinstance(obj, list): return [clean_data(i) for i in obj]
    return obj

# --- CONFIGURAÇÕES ---
SYMBOL = 'BTC/USDT'
TIMEFRAME = '5m'
MODEL_PATH = "models/sniper_pro_finished"
DATA_FILE = "data/live_market_data.csv"
INITIAL_BALANCE = 800 
FEE_RATE = 0.0005

# --- RISCO ---
STOP_LOSS_PCT = 0.015    
TAKE_PROFIT_PCT = 0.035  
TRAILING_TRIGGER = 0.010 
COOLDOWN_SECONDS = 300   

state = {
    "balance": INITIAL_BALANCE,
    "position": 0,
    "entry_price": 0.0,
    "entry_time": None,
    "pnl_open": 0.0,
    "status": "INICIANDO SISTEMA...", # Status inicial
    "chart_data": [],       
    "last_candle": {},
    "trades_history": [],
    "stats": { "wins": 0, "losses": 0, "win_rate": 0.0, "total_trades": 0, "start_time": datetime.now().timestamp() },
    "training": { "is_training": False, "last_evolution": "Aguardando", "generation": 1 }
}

model_lock = threading.Lock()
# ATENÇÃO: Definimos model como None aqui para não travar o início
model = None 
dummy_env = None

print("--- SNIPER PRO V6 (CLOUD OPTIMIZED) ---")

if not os.path.exists("data"): os.makedirs("data")

def calculate_features(df):
    try:
        df = df.copy()
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
        df = df.dropna()
        cols = ['log_ret', 'rsi', 'rsi_slope', 'macd_diff', 'bb_pband', 'bb_width', 'dist_ema50', 'dist_ema200', 'atr_pct']
        return df[cols], df
    except:
        return None, None

# --- FUNÇÃO DE CARREGAMENTO DO MODELO (ULTRA OTIMIZADA) ---
def load_brain_logic():
    global model, dummy_env, state
    print(">>> 🧹 LIMPANDO MEMÓRIA ANTES DE CARREGAR A IA...")
    
    # Força limpeza do lixo da memória RAM
    gc.collect()
    
    print(">>> CARREGANDO CÉREBRO EM BACKGROUND... (Modo Econômico)")
    try:
        # Cria um DataFrame minúsculo apenas para inicializar o ambiente
        # Reduzimos de 200 para 50 linhas para gastar menos RAM na criação
        dummy_df = pd.DataFrame({'close': [100]*50}) 
        cols = ['log_ret', 'rsi', 'rsi_slope', 'macd_diff', 'bb_pband', 'bb_width', 'dist_ema50', 'dist_ema200', 'atr_pct']
        for c in cols: dummy_df[c] = 0.0
        
        dummy_env = BitcoinTradingEnv(dummy_df)
        
        # Carrega o modelo forçando uso de CPU e evitando ocupar VRAM (que não existe)
        model = RecurrentPPO.load(MODEL_PATH, env=dummy_env, device="cpu")
        
        print(">>> CÉREBRO CARREGADO COM SUCESSO! 🧠")
        state["status"] = "NEUTRO (IA PRONTA)"
        
        # Limpa novamente vestígios do processo de carregamento
        del dummy_df
        gc.collect()
        
    except Exception as e:
        print(f"ERRO CRÍTICO (MEMÓRIA?): {e}")
        state["status"] = "ERRO: MEMÓRIA INSUFICIENTE"
        model = None

# --- ENGINE DE AUTO-TREINAMENTO ---
def train_brain_background():
    global model, state
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}] 🧬 INICIANDO EVOLUÇÃO NEURAL...")
    state["training"]["is_training"] = True
    
    try:
        if not os.path.exists(DATA_FILE): return
        df = pd.read_csv(DATA_FILE)
        if len(df) < 200:
            state["training"]["is_training"] = False
            return

        df_processed = calculate_features(df)
        train_env = BitcoinTradingEnv(df_processed)
        learner_model = RecurrentPPO.load(MODEL_PATH, env=train_env)
        learner_model.learn(total_timesteps=4096) 
        learner_model.save(MODEL_PATH)
        
        with model_lock:
            model = RecurrentPPO.load(MODEL_PATH, env=dummy_env)
            state["training"]["generation"] += 1
            state["training"]["last_evolution"] = timestamp
            
        print(f"[{timestamp}] ✅ EVOLUÇÃO CONCLUÍDA! Geração {state['training']['generation']} Ativa.")
        
    except Exception as e:
        print(f"❌ ERRO NA EVOLUÇÃO: {e}")
    
    state["training"]["is_training"] = False

def trigger_evolution():
    t = threading.Thread(target=train_brain_background)
    t.start()

def save_learning_data(ohlcv):
    try:
        df_new = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        if not os.path.exists(DATA_FILE):
            df_new.to_csv(DATA_FILE, index=False)
        else:
            df_old = pd.read_csv(DATA_FILE)
            df_combined = pd.concat([df_old, df_new])
            df_combined = df_combined.drop_duplicates(subset=['timestamp'])
            if len(df_combined) > 10000: df_combined = df_combined.iloc[-10000:]
            df_combined.to_csv(DATA_FILE, index=False)
    except: pass

async def sniper_loop():
    global state, model
    
    # --- AQUI ESTÁ O TRUQUE PARA O RENDER ---
    # Primeiro esperamos o servidor ligar, depois carregamos o modelo
    await asyncio.sleep(2) 
    load_brain_logic() 
    # ----------------------------------------

    exchange = ccxt.binance({'enableRateLimit': True, 'options': {'defaultType': 'future'}})
    lstm_states = None
    episode_starts = np.ones((1,), dtype=bool)
    last_exit_time = 0 
    highest_pnl_pct = -1.0 
    
    print(">>> LOOP DE TRADING INICIADO...")
    
    while True:
        try:
            # Se o modelo ainda não carregou, espera e mostra no status
            if model is None:
                state["status"] = "CARREGANDO CÉREBRO..."
                await asyncio.sleep(5)
                continue

            ohlcv = await exchange.fetch_ohlcv(SYMBOL, timeframe=TIMEFRAME, limit=200)
            save_learning_data(ohlcv) 

            # Processamento de Dados
            formatted_history = []
            for candle in ohlcv:
                formatted_history.append({
                    "time": int(candle[0] / 1000), "open": float(candle[1]), "high": float(candle[2]), "low": float(candle[3]), "close": float(candle[4])
                })
            state["chart_data"] = formatted_history
            state["last_candle"] = formatted_history[-1]
            
            df_raw = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            current_price = float(df_raw.iloc[-1]['close'])
            timestamp_now_str = datetime.now().strftime("%H:%M:%S")
            now_ts = datetime.now().timestamp()
            
            # PnL Calc
            current_pnl_pct = 0.0
            if state["position"] != 0:
                diff = (current_price - state["entry_price"]) / state["entry_price"]
                current_pnl_pct = diff if state["position"] == 1 else -diff
                state["pnl_open"] = state["balance"] * current_pnl_pct
                if current_pnl_pct > highest_pnl_pct: highest_pnl_pct = current_pnl_pct
            else:
                state["pnl_open"] = 0.0

            # Cooldown
            time_since_exit = now_ts - last_exit_time
            if state["position"] == 0 and time_since_exit < COOLDOWN_SECONDS:
                remaining = int(COOLDOWN_SECONDS - time_since_exit)
                state["status"] = f"AGUARDANDO ({remaining}s)"
                await asyncio.sleep(1) 
                continue 

            # GESTÃO DE RISCO
            forced_exit = False
            exit_reason = ""
            if state["position"] != 0:
                if current_pnl_pct <= -STOP_LOSS_PCT: forced_exit, exit_reason = True, "STOP LOSS 🛑"
                elif current_pnl_pct >= TAKE_PROFIT_PCT: forced_exit, exit_reason = True, "TAKE PROFIT 🎯"
                elif highest_pnl_pct >= TRAILING_TRIGGER and current_pnl_pct <= (highest_pnl_pct - 0.005): forced_exit, exit_reason = True, "TRAILING STOP 👻"

            if forced_exit:
                final_pnl = (state["balance"] * current_pnl_pct) - (state["balance"] * FEE_RATE)
                state["balance"] += final_pnl
                
                is_win = final_pnl > 0
                state["trades_history"].insert(0, {
                    "id": len(state["trades_history"]), "type": "COMPRA" if state["position"] == 1 else "VENDA",
                    "entry_time": state["entry_time"], "exit_time": timestamp_now_str,
                    "entry_price": state["entry_price"], "exit_price": current_price, "pnl": final_pnl, "result": exit_reason
                })
                if is_win: state["stats"]["wins"] += 1
                else: state["stats"]["losses"] += 1
                total = state["stats"]["wins"] + state["stats"]["losses"]
                state["stats"]["win_rate"] = (state["stats"]["wins"] / total * 100) if total > 0 else 0

                print(f"[{timestamp_now_str}] SAÍDA: {exit_reason} | PnL: {final_pnl:.2f}")
                state["position"] = 0
                state["status"] = "COOLDOWN ❄️"
                last_exit_time = now_ts
                if not state["training"]["is_training"]: trigger_evolution()
                continue

            # PREDIÇÃO DA IA
            features, full_df = calculate_features(df_raw)
            if model and features is not None:
                ema200, rsi = full_df.iloc[-1]['ema200'], full_df.iloc[-1]['rsi']
                obs = features.iloc[-1].values
                
                with model_lock:
                    action, lstm_states = model.predict(obs, state=lstm_states, episode_start=episode_starts, deterministic=True)
                
                episode_starts = np.zeros((1,), dtype=bool)
                act_idx = int(action.item()) if isinstance(action, np.ndarray) else int(action)
                target_pos = 1 if act_idx == 1 else (-1 if act_idx == 2 else 0)
                
                if target_pos == 1 and current_price < ema200: target_pos = 0
                if target_pos == 1 and rsi > 70: target_pos = 0

                if target_pos != state["position"]:
                    if state["position"] != 0: # Fechamento por IA
                        final_pnl = (state["balance"] * current_pnl_pct) - (state["balance"] * FEE_RATE)
                        state["balance"] += final_pnl
                        is_win = final_pnl > 0
                        state["trades_history"].insert(0, {
                            "id": len(state["trades_history"]), "type": "COMPRA" if state["position"] == 1 else "VENDA",
                            "entry_time": state["entry_time"], "exit_time": timestamp_now_str,
                            "entry_price": state["entry_price"], "exit_price": current_price, "pnl": final_pnl, "result": "IA DECIDIU"
                        })
                        if is_win: state["stats"]["wins"] += 1
                        else: state["stats"]["losses"] += 1
                        total = state["stats"]["wins"] + state["stats"]["losses"]
                        state["stats"]["win_rate"] = (state["stats"]["wins"] / total * 100) if total > 0 else 0
                        
                        print(f"[{timestamp_now_str}] 🤖 IA FECHOU. PnL: {final_pnl:.2f}")
                        state["position"] = 0
                        state["status"] = "COOLDOWN ❄️"
                        last_exit_time = now_ts
                        if not state["training"]["is_training"]: trigger_evolution()
                        continue 

                    if target_pos != 0:
                        state["balance"] -= state["balance"] * FEE_RATE
                        state["entry_price"] = current_price
                        state["entry_time"] = timestamp_now_str
                        state["position"] = target_pos
                        state["status"] = "COMPRADO 🟢" if target_pos == 1 else "VENDIDO 🔴"
                        highest_pnl_pct = -1.0
                        print(f"[{timestamp_now_str}] 🚀 IA ABRIU {state['status']}")
            
            await asyncio.sleep(2)

        except Exception as e:
            print(f"❌ ERRO: {e}")
            await asyncio.sleep(5)

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            uptime = int(datetime.now().timestamp() - state["stats"]["start_time"])
            payload = state.copy()
            payload["uptime"] = f"{uptime//3600:02}:{(uptime%3600)//60:02}:{uptime%60:02}"
            await websocket.send_json(clean_data(payload))
            await asyncio.sleep(1)
    except: pass

@app.on_event("startup")
async def startup_event():
    # Iniciamos a tarefa em background, permitindo que a API inicie imediatamente
    asyncio.create_task(sniper_loop())