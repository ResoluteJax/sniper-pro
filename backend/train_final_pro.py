import os
import pandas as pd
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.monitor import Monitor
from envs.trading_env import BitcoinTradingEnv

# --- CONFIGURAÇÕES ---
N_ENVS = 8  # Seus 8 núcleos trabalhando
TOTAL_TIMESTEPS = 3_000_000

# Hiperparâmetros
# Nota: A Learning Rate será ajustada automaticamente se carregarmos um modelo antigo
HYPERPARAMS = {
    'learning_rate': 0.0003,      
    'gamma': 0.90,                
    'ent_coef': 0.1,              
    'batch_size': 512,            
    'n_steps': 512,               
    'policy_kwargs': dict(
        net_arch=[64, 64],        
        lstm_hidden_size=64,
        n_lstm_layers=1,
        shared_lstm=True,
        enable_critic_lstm=False
    ),
    'device': 'cpu'               
}

# --- FUNÇÃO FABRICA DE AMBIENTES ---
def make_env(rank, seed=0):
    def _init():
        # Lê o CSV PRO
        df = pd.read_csv('btc_futures_data_PRO.csv')
        df = df.dropna().reset_index(drop=True)
        
        # Cria o ambiente (Vai pegar o fee_rate=0.0001 que você definiu no arquivo)
        env = BitcoinTradingEnv(df) 
        env = Monitor(env) 
        env.reset(seed=seed + rank)
        return env
    set_random_seed(seed)
    return _init

# --- BLOCO PRINCIPAL ---
if __name__ == '__main__':
    # Pastas
    save_dir = "./models/final_pro/"
    log_dir = "./tensorboard_logs/"
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    print(f"--- INICIANDO TREINO DE REFINO ({N_ENVS} Núcleos) ---")
    
    # 1. Criação dos Ambientes
    env = SubprocVecEnv([make_env(i) for i in range(N_ENVS)])

    # 2. Callback
    checkpoint_callback = CheckpointCallback(
        save_freq=100000 // N_ENVS, 
        save_path=save_dir,
        name_prefix='sniper_pro'
    )

    # 3. Carregamento Inteligente (Transfer Learning)
    model_path = f"{save_dir}/sniper_pro_finished.zip"
    
    if os.path.exists(model_path):
        print(f"\n>>> MODELO VETERANO ENCONTRADO: {model_path}")
        print(">>> Carregando pesos e iniciando O DESAFIO FINAL (Taxas Reais)...")
        
        model = RecurrentPPO.load(model_path, env=env)
        
        # ULTRA FINE TUNING: 
        # Reduzimos para 5e-5 (0.00005). Aprendizado microscópico.
        # Ele só vai alterar os neurônios se tiver MUITA certeza.
        model.learning_rate = 0.00005 
        
        print(f">>> Learning Rate cirúrgica: {model.learning_rate}")
        print(">>> Objetivo: Sobreviver no mercado real.")
        
    else:
        print("\n[!] Modelo anterior NÃO encontrado.")
        print("[!] Iniciando treinamento do ZERO (Crazy Scalper Mode).")
        model = RecurrentPPO(
            "MlpLstmPolicy",
            env,
            verbose=1,
            tensorboard_log=log_dir,
            **HYPERPARAMS
        )

    # 4. Treinamento
    print("\n--- GO! (Acompanhe o FPS) ---")
    try:
        model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=checkpoint_callback)
        model.save(f"{save_dir}/sniper_pro_finished")
        print("--- TREINO DE REFINO FINALIZADO ---")
    except KeyboardInterrupt:
        print("\nTreino interrompido. Salvando...")
        model.save(f"{save_dir}/sniper_pro_interrupted")
    finally:
        env.close()