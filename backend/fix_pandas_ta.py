import os

# --- CAMINHO DIRETO (Sem importar a biblioteca quebrada) ---
# Assume que estamos na raiz do projeto 'Sniper_Final' e o venv se chama 'venv_live'
base_dir = os.getcwd()
path = os.path.join(base_dir, "venv_live", "Lib", "site-packages", "pandas_ta")

print(f"🏥 Iniciando cirurgia em: {path}")

if not os.path.exists(path):
    print(f"❌ ERRO: Não encontrei a pasta em: {path}")
    print("Verifique se você está na pasta C:\\Projetos\\Sniper_Final")
    exit()

files_fixed = 0
errors = 0

# Varre todos os arquivos e subpastas
for root, dirs, files in os.walk(path):
    for file in files:
        if file.endswith(".py"):
            filepath = os.path.join(root, file)
            try:
                # Lê o arquivo
                with open(filepath, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Se encontrar o código antigo, substitui
                # Procura por 'NaN' e troca por 'nan'
                if "from numpy import NaN" in content:
                    new_content = content.replace("from numpy import NaN", "from numpy import nan")
                    
                    with open(filepath, 'w', encoding='utf-8') as f:
                        f.write(new_content)
                    
                    print(f"✅ Corrigido: {file}")
                    files_fixed += 1
            except Exception as e:
                print(f"❌ Erro em {file}: {e}")
                errors += 1

print(f"\nRESUMO: {files_fixed} arquivos corrigidos.")
print("Agora tente rodar o robô!")