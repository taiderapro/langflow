from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

class Chatbot:
    def __init__(self, api_key, vector_store=None):
        """Inicializa o Chatbot com o modelo de linguagem e armazenamento vetorial opcional."""
        self.api_key = api_key
        self.vector_store = vector_store
        self.model = ChatOpenAI(openai_api_key=self.api_key, model="gpt-3.5-turbo", request_timeout=30)
    
    def generate_lesson_plan(self, topic, level, duration):
        """Gera um plano de aula baseado nos parâmetros fornecidos."""
        try:
            # Criação do prompt template
            prompt_template = PromptTemplate(
                input_variables=["topic", "level", "duration"],
                template=(
                    "Crie um plano de aula com o seguinte tema: {topic}.\n"
                    "Nível dos alunos: {level}.\n"
                    "Duração: {duration}.\n"
                    "Formate o plano de aula de forma clara e organizada."
                ),
            )
            
            # Inicialização da cadeia LLM
            chain = LLMChain(llm=self.model, prompt=prompt_template)
            
            # Execução da cadeia para gerar o plano de aula
            response = chain.run({"topic": topic, "level": level, "duration": duration})
            return response
        except Exception as e:
            raise AttributeError(f"Erro ao gerar o plano de aula: {str(e)}")
