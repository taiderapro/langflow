from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter

def create_vector_store(documents, api_key):
    """Cria um armazenamento vetorial a partir de documentos."""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_documents(documents)

    embeddings = OpenAIEmbeddings(api_key=api_key)
    vector_store = FAISS.from_documents(texts, embeddings)
    return vector_store
