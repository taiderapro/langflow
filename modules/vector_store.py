import openai
from langchain.vectorstores import FAISS
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
        embeddings = []
        for text in texts:
            response = openai.Embedding.create(
                input=text,
                model=model
            )
            embeddings.append(response["data"][0]["embedding"])

        # Criar o armazenamento vetorial
        vector_store = FAISS.from_texts(texts, embeddings)
        return vector_store

    except Exception as e:
        logger.error(f"Erro ao criar o armazenamento vetorial: {e}")
        raise RuntimeError(f"Erro ao criar o armazenamento vetorial: {e}")
