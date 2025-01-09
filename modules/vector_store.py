import openai
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.docstore.document import Document
import logging

logger = logging.getLogger(__name__)

def create_vector_store(documents, api_key, model="text-embedding-3-small"):
    """
    Cria o armazenamento vetorial usando embeddings da OpenAI.
    """
    try:
        # Configurar a API da OpenAI
        openai.api_key = api_key

        # Processar documentos e gerar embeddings
        texts = [doc.page_content for doc in documents]
        metadata = [doc.metadata for doc in documents]

        # Chamar a API da OpenAI para gerar embeddings
        embeddings = OpenAIEmbeddings(openai_api_key=api_key, model=model)
        vector_store = FAISS.from_texts(texts, embeddings, metadatas=metadata)

        logger.info("Armazenamento vetorial criado com sucesso.")
        return vector_store

    except Exception as e:
        logger.error(f"Erro ao criar o armazenamento vetorial: {e}", exc_info=True)
        raise RuntimeError(f"Erro ao criar o armazenamento vetorial: {e}")
