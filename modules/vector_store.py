from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from modules.logger import get_logger

logger = get_logger(__name__)

def create_vector_store(documents, api_key):
    """
    Cria um armazenamento vetorial usando FAISS com os documentos fornecidos.

    Args:
        documents (list): Lista de objetos `Document`.
        api_key (str): Chave de API para o OpenAI.

    Returns:
        FAISS: Armazenamento vetorial criado.

    Raises:
        RuntimeError: Se ocorrer algum erro durante o processo.
    """
    try:
        # Inicializar embeddings do OpenAI
        logger.info("Inicializando embeddings do OpenAI.")
        embeddings = OpenAIEmbeddings(openai_api_key=api_key)

        # Dividir documentos em chunks menores
        logger.info("Dividindo documentos em chunks.")
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        chunks = []

        for doc in documents:
            if isinstance(doc, Document):
                split_docs = text_splitter.split_documents([doc])
                chunks.extend(split_docs)
                logger.info(f"Documento dividido em {len(split_docs)} chunks.")
            else:
                logger.warning("Objeto inválido encontrado na lista de documentos.")

        if not chunks:
            raise ValueError("Nenhum chunk foi gerado a partir dos documentos.")

        # Criar armazenamento vetorial
        logger.info("Criando armazenamento vetorial com FAISS.")
        vector_store = FAISS.from_documents(chunks, embeddings)
        logger.info("Armazenamento vetorial criado com sucesso.")
        return vector_store

    except Exception as e:
        logger.error("Erro ao criar o armazenamento vetorial.", exc_info=True)
        raise RuntimeError(f"Erro ao criar o armazenamento vetorial: {str(e)}")
