import streamlit as st
from modules.logger import get_logger
from langchain.schema import SystemMessage, HumanMessage, AIMessage
from langchain.chat_models import ChatOpenAI
from modules.chat import Chatbot
from modules.file_loader import process_uploaded_files
from modules.vector_store import create_vector_store
from docx import Document
from io import BytesIO

# Configuração inicial do Streamlit
st.set_page_config(page_title="Agente de IA para Professores", layout="wide")

# Inicializar logger
logger = get_logger(__name__)

# Configurações de API
try:
    api_key = st.secrets["OPENAI_API_KEY"]
except KeyError:
    st.error("Por favor, configure a variável OPENAI_API_KEY nos segredos do Streamlit.")
    st.stop()

# Upload de arquivos
st.sidebar.header("Upload de Arquivos")
uploaded_files = st.sidebar.file_uploader(
    "Carregue arquivos (PDF, DOCX, TXT)", type=["pdf", "docx", "txt"], accept_multiple_files=True
)

documents = []
if uploaded_files:
    try:
        documents = process_uploaded_files(uploaded_files)
        st.sidebar.success("Arquivos processados com sucesso!")
    except ValueError as e:
        st.sidebar.error(str(e))
        logger.error("Erro ao processar arquivos", exc_info=True)

# Criar armazenamento vetorial
# Criar armazenamento vetorial com novo modelo de embeddings
vector_store = None
if documents:
    try:
        vector_store = create_vector_store(
            documents, api_key, model="text-embedding-3-small"  # Modelo da OpenAI
        )
        st.success("Armazenamento vetorial criado com sucesso!")
    except Exception as e:
        logger.error("Erro ao criar o armazenamento vetorial", exc_info=True)
        st.error("Erro ao criar armazenamento vetorial. Consulte o log para mais detalhes.")

# Inicializar o chatbot
try:
    chatbot = ChatOpenAI(api_key=api_key, model="gpt-3.5-turbo", request_timeout=30)
except Exception as e:
    logger.error(f"Erro ao conectar com o modelo OpenAI: {str(e)}", exc_info=True)
    st.error(f"Erro ao conectar com o modelo OpenAI: {str(e)}")
    st.stop()

# Função de exportação para DOCX
def export_to_docx(lesson_plan, file_name="Plano_de_Aula.docx"):
    document = Document()
    document.add_heading("Plano de Aula", level=1)
    for line in lesson_plan.split("\n"):
        document.add_paragraph(line)
    docx_buffer = BytesIO()
    document.save(docx_buffer)
    docx_buffer.seek(0)
    return docx_buffer

# Chatbot interativo
st.header("Chatbot Interativo")

if "chat_submitted" not in st.session_state:
    st.session_state.chat_submitted = False
if "messages" not in st.session_state:
    st.session_state.messages = []

user_input = st.text_input("Digite sua pergunta:")
chat_submitted = st.button("Enviar Mensagem")

if chat_submitted and user_input:
    st.session_state.chat_submitted = True
    st.session_state.messages.append(HumanMessage(content=user_input))

if st.session_state.chat_submitted:
    if vector_store:
        from langchain.chains import RetrievalQA
        retriever = vector_store.as_retriever()
        qa_chain = RetrievalQA.from_chain_type(llm=chatbot.model, retriever=retriever)
        response = qa_chain.run(user_input)
    else:
        response = chatbot.respond(st.session_state.messages)

    st.session_state.messages.append(AIMessage(content=response))
    st.session_state.chat_submitted = False

for msg in st.session_state.messages:
    st.write(f"**{msg.__class__.__name__}:** {msg.content}")

# Geração de Planos de Aula Personalizados
st.header("Criação de Planos de Aula Personalizados")

if "lesson_plan" not in st.session_state:
    st.session_state.lesson_plan = None
if "form_submitted" not in st.session_state:
    st.session_state.form_submitted = False

if st.button("Gerar Nova Aula"):
    chatbot.reset_lesson_memory()
    st.session_state.lesson_plan = None
    st.session_state.form_submitted = False
    st.success("Memória de aula resetada. Pronto para uma nova aula!")

with st.form(key="lesson_plan_form"):
    topic = st.text_input("Tema da aula", placeholder="Exemplo: Direito Imobiliário")
    level = st.selectbox("Nível dos alunos", ["Fundamental", "Médio", "Superior"])
    duration = st.text_input("Duração da aula", placeholder="Exemplo: 1 hora")
    submitted = st.form_submit_button("Gerar Plano de Aula")

    if submitted:
        st.session_state.form_submitted = True

if st.session_state.form_submitted:
    if topic and level and duration:
        try:
            lesson_plan = chatbot.generate_lesson_plan(topic, level, duration)
            st.session_state.lesson_plan = lesson_plan
            st.subheader("Plano de Aula Gerado:")
            st.write(lesson_plan)
        except Exception as e:
            logger.error(f"Erro ao gerar plano de aula: {str(e)}", exc_info=True)
            st.error(f"Erro ao gerar plano de aula: {str(e)}")
    else:
        st.warning("Por favor, preencha todos os campos antes de gerar o plano de aula.")
