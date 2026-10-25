
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_core.documents import Document

# Модель для создания эмбеддингов
embeddings = OllamaEmbeddings(
    model="nomic-embed-text"
)

# Документы базы знаний
documents = [
    Document(
        page_content=(
            "Битрикс24 - платформа для управления бизнесом. "
            "Включает CRM, управление задачами и проектами, "
            "командные коммуникации и автоматизацию "
            "бизнес-процессов."
        ),
        metadata={"company": "Битрикс24"}
    ),
    Document(
        page_content=(
            "ПланФикс - система управления проектами, "
            "задачами и работой с клиентами. "
            "Поддерживает автоматизацию процессов "
            "и настройку рабочих инструментов "
            "под потребности разных отделов."
        ),
        metadata={"company": "ПланФикс"}
    ),
    Document(
        page_content=(
            "FlowSpace - разрабатываемая платформа "
            "для управления задачами и проектами. "
            "Предусмотрены Kanban-доски, напоминания, "
            "контроль сроков и автоматизация процессов."
        ),
        metadata={"company": "FlowSpace"}
    )
]

# Создаём векторную базу данных
vector_store = Chroma.from_documents(
    documents=documents,
    embedding=embeddings,
    persist_directory="./chroma_db",
    collection_name="competitors"
)

print("База знаний успешно создана!")

# Проверяем смысловой поиск
results = vector_store.similarity_search(
    "Какие возможности есть у ПланФикса?",
    k=1
)

print("\nРЕЗУЛЬТАТ ПОИСКА:")
for doc in results:
    print(doc.page_content)
