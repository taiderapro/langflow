from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

class Chatbot:
    def __init__(self, api_key, vector_store=None):
        self.api_key = api_key
        self.vector_store = vector_store
        self.model = ChatOpenAI(api_key=self.api_key, model="gpt-3.5-turbo", request_timeout=30)
    
    def generate_lesson_plan(self, topic, level, duration):
        """Gera um plano de aula baseado nos parâmetros fornecidos."""
        try:
            prompt = f"""
            Crie um plano de aula com o seguinte tema: {topic}.
            Nível dos alunos: {level}.
            Duração: {duration}.
            Formate o plano de aula de forma clara e organizada.
            """
            response = self.model(prompt)
            return response["choices"][0]["message"]["content"]
        except Exception as e:
            raise AttributeError(f"Erro ao gerar o plano de aula: {str(e)}")

