import logging

# Configuração do logger
logging.basicConfig(
    filename="app_logs.log",  # Nome do arquivo de log
    level=logging.ERROR,  # Nível de log (ERROR para capturar erros)
    format="%(asctime)s - %(levelname)s - %(message)s",
)

try:
    # Exemplo de operação que pode gerar um erro
    vector_store = create_vector_store(documents, api_key)
except Exception as e:
    logging.error("Erro ao criar armazenamento vetorial", exc_info=True)  # Log detalhado
    st.error("Erro ao criar armazenamento vetorial. Consulte o log para mais detalhes.")
