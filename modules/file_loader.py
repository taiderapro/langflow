import os
from langchain.document_loaders import PyPDFLoader, TextLoader
from langchain.schema import Document
from docx import Document as DocxDocument
from tempfile import NamedTemporaryFile


def save_to_temp_file(uploaded_file):
    """Salva o arquivo enviado pelo Streamlit em um arquivo temporário."""
    temp_file = NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1])
    temp_file.write(uploaded_file.read())
    temp_file.close()
    return temp_file.name


def load_pdf(file_path):
    """Carrega e processa arquivos PDF."""
    loader = PyPDFLoader(file_path)
    return loader.load()


def load_docx(file_path):
    """Carrega e processa arquivos DOCX."""
    doc = DocxDocument(file_path)
    text = "\n".join([p.text for p in doc.paragraphs if p.text.strip()])
    return [Document(page_content=text)]


def load_txt(file_path):
    """Carrega e processa arquivos TXT."""
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()
    return [Document(page_content=content)]


def process_uploaded_files(uploaded_files):
    """Processa os arquivos enviados pelo usuário e retorna uma lista de documentos."""
    documents = []
    for uploaded_file in uploaded_files:
        try:
            # Salva o arquivo temporariamente
            temp_file_path = save_to_temp_file(uploaded_file)

            # Identifica o tipo de arquivo e processa
            if uploaded_file.name.endswith(".pdf"):
                documents.extend(load_pdf(temp_file_path))
            elif uploaded_file.name.endswith(".docx"):
                documents.extend(load_docx(temp_file_path))
            elif uploaded_file.name.endswith(".txt"):
                documents.extend(load_txt(temp_file_path))
            else:
                raise ValueError(f"Formato de arquivo não suportado: {uploaded_file.name}")

            # Remove o arquivo temporário
            os.remove(temp_file_path)

        except Exception as e:
            raise RuntimeError(f"Erro ao processar o arquivo {uploaded_file.name}: {str(e)}") from e

    return documents
