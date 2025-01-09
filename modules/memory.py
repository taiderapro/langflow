from langchain.memory import ConversationBufferMemory

def get_memory():
    """Retorna uma instância de memória para o chatbot."""
    return ConversationBufferMemory(return_messages=True)
