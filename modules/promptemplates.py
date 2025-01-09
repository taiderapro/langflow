from langchain.prompts import PromptTemplate

lesson_plan_prompt = PromptTemplate(
    input_variables=["topic", "level", "duration"],
    template=(
        "Você é um assistente que ajuda professores a criar aulas personalizadas."
        "Crie um plano de aula sobre o tema '{topic}', para alunos do nível '{level}', com duração de '{duration}'."
        "Inclua objetivos da aula, uma introdução, os tópicos principais e uma atividade final."
    )
)
