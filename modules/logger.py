import logging
import os

# Diretório onde o log será salvo
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)  # Cria o diretório de logs, se não existir

# Caminho do arquivo de log
LOG_FILE = os.path.join(LOG_DIR, "app.log")

# Configuração básica do logger
logging.basicConfig(
    filename=LOG_FILE,
    level=logging.DEBUG,  # Altere para INFO, WARNING, ERROR conforme necessário
    format="%(asctime)s - %(levelname)s - %(message)s",
)

# Função para obter o logger configurado
def get_logger(name):
    return logging.getLogger(name)
