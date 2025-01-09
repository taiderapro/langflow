from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory

class LessonPlanner:
    def __init__(self, llm, vector_store=None):
        self.llm = llm
        self.vector_store = vector_store
        self.memory = ConversationBufferMemory(return_messages=True)

    def retrieve_context(self, topic):
        """Busca contexto no armazenamento vetorial baseado no tema."""
        if self.vector_store:
            retriever = self.vector_store.as_retriever()
            results = retriever.get_relevant_documents(topic)
            return "\n".join([doc.page_content for doc in results])
        return ""

    def generate_lesson_plan(self, topic, level, duration, chat_context):
        """Gera o plano de aula considerando o chat e o vetor."""
        vector_context = self.retrieve_context(topic)
        combined_context = f"{vector_context}\n\n{chat_context}"

        # Ajustando para funcionar com múltiplos parâmetros
        lesson_plan_prompt = PromptTemplate(
            input_variables=["topic", "level", "duration", "context"],
            template=(
                "Você é um assistente que ajuda professores a criar aulas personalizadas. "
                "Aqui está o contexto adicional: {context}\n\n"
                "Crie um plano de aula sobre o tema '{topic}', para alunos do nível '{level}', com duração de '{duration}'. "
                "Inclua objetivos da aula, uma introdução, os tópicos principais e uma atividade final."
            )
        )

        chain = LLMChain(llm=self.llm, prompt=lesson_plan_prompt)
        return chain.run({
            "topic": topic,
            "level": level,
            "duration": duration,
            "context": combined_context
        })

    def reset_memory(self):
        """Reseta a memória do gerador de aulas."""
        self.memory = ConversationBufferMemory(return_messages=True)
