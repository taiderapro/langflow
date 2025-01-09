from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.docstore.document import Document
import numpy as np

def create_vector_store(documents, api_key, model="text-embedding-3-small"):
    """
    Cria um armazenamento vetorial usando os modelos de embeddings da OpenAI.
    
    Args:
        documents (List[Document]): Lista de documentos para serem processados.
        api_key (str): Chave da API da OpenAI.
        model (str): Modelo de embeddings da OpenAI. Padrão: "text-embedding-3-small".
    
    Returns:
        FAISS: Armazenamento vetorial criado.
    """
    try:
        # Inicializar embeddings da OpenAI
        embeddings = OpenAIEmbeddings(openai_api_key=api_key, model=model)
        
        # Criar o armazenamento vetorial com FAISS
        vector_store = FAISS.from_documents(documents, embeddings)
        return vector_store
    except Exception as e:
        raise RuntimeError(f"Erro ao criar o armazenamento vetorial: {str(e)}")


def normalize_embeddings(vector, norm_type="l2"):
    """
    Normaliza um vetor de embeddings.
    
    Args:
        vector (list): Embedding a ser normalizado.
        norm_type (str): Tipo de normalização. Padrão: "l2".
    
    Returns:
        np.array: Embedding normalizado.
    """
    x = np.array(vector)
    if norm_type == "l2":
        norm = np.linalg.norm(x)
        if norm == 0:
            return x
        return x / norm
    return x
