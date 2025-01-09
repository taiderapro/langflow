from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS

def create_vector_store(documents, api_key):
    """Cria um armazenamento vetorial com base nos documentos fornecidos."""
    try:
        # Dividir documentos em chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,  # Tamanho do chunk (em caracteres)
            chunk_overlap=100,  # Sobreposição entre chunks para contexto
        )
        split_documents = text_splitter.split_documents(documents)

        # Criar embeddings e armazenamento vetorial
        embeddings = OpenAIEmbeddings(openai_api_key=api_key)
        vector_store = FAISS.from_documents(split_documents, embeddings)
        return vector_store
    except Exception as e:
        raise RuntimeError(f"Erro ao criar o armazenamento vetorial: {str(e)}")
