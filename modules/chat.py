from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

class Chatbot:
    def __init__(self, api_key, vector_store=None):
        self.api_key = api_key
        self.vector_store = vector_store
        self.llm = ChatOpenAI(api_key=self.api_key, model="gpt-3.5-turbo", request_timeout=30)

    def generate_lesson_plan(self, topic, level, duration):
        """Gera um plano de aula personalizado."""
        try:
            # Template para a geração de plano de aula
            lesson_plan_prompt = PromptTemplate(
                input_variables=["topic", "level", "duration"],
                template=(
                    "Crie um plano de aula sobre o tema '{topic}' para alunos de nível {level}. "
                    "A aula deve ter a duração de {duration}. Inclua objetivos, introdução, tópicos principais e atividades finais."
                )
            )

            # Criar o LLMChain para a geração de plano de aula
            chain = LLMChain(llm=self.llm, prompt=lesson_plan_prompt)

            # Executar o chain
            lesson_plan = chain.run({
                "topic": topic,
                "level": level,
                "duration": duration,
            })

            return lesson_plan
        except Exception as e:
            raise RuntimeError(f"Erro ao gerar o plano de aula: {str(e)}")
