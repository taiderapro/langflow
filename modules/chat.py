from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from modules.lesson_planner import LessonPlanner

class Chatbot:
    def __init__(self, api_key, vector_store=None):
        self.memory = ConversationBufferMemory(return_messages=True)
        self.model = ChatOpenAI(api_key=api_key, model="gpt-4")
        self.lesson_planner = LessonPlanner(llm=self.model, vector_store=vector_store)

    def respond(self, messages):
        """Responde a mensagens gerais e armazena no contexto."""
        for msg in messages:
            self.memory.chat_memory.add_message(msg)
        response = self.model.invoke(self.memory.chat_memory.messages)
        return response.content

    def generate_lesson_plan(self, topic, level, duration):
        """Cria um plano de aula considerando o contexto do chat."""
        chat_context = "\n".join(
            [f"{msg.role}: {msg.content}" for msg in self.memory.chat_memory.messages]
        )
        return self.lesson_planner.generate_lesson_plan(topic, level, duration, chat_context)

    def reset_lesson_memory(self):
        """Reseta a memória do LessonPlanner."""
        self.lesson_planner.reset_memory()

