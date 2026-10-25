from typing import TypedDict

from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver

from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

import asyncio
import sys

from langchain_mcp_adapters.client import MultiServerMCPClient

# языковая модель ollama
model = ChatOllama(model="qwen3:8b", temperature=0)

# пдключаем готовую базу знаний
embeddings = OllamaEmbeddings(
    model="nomic-embed-text"
)

vector_store = Chroma(
    collection_name="competitors",
    embedding_function=embeddings,
    persist_directory="./chroma_db"
)


# получение содержимого сайта через MCP-сервер
async def fetch_website(url):
    client = MultiServerMCPClient(
        {
            "fetch": {
                "transport": "stdio",
                "command": sys.executable,
                "args": ["-m", "mcp_server_fetch"]
            }
        }
    )

    tools = await client.get_tools()

    fetch_tool = next(
        tool for tool in tools
        if tool.name == "fetch"
    )

    result = await fetch_tool.ainvoke({
        "url": url
    })

    return str(result)



# поиск информации в базе знаний
def search_knowledge(company):
    documents = vector_store.similarity_search(
        query=f"Функции и возможности системы {company}",
        k=1,
        filter={"company": company}
    )

    return "\n".join(doc.page_content for doc in documents)


# состояние графа
class AgentState(TypedDict):
    all_analyses: str
    recommendations: str




def competitor_agent(state: AgentState):
    all_analyses = []

    websites = {
        "Битрикс24": "https://www.bitrix24.ru/",
        "ПланФикс": "https://planfix.ru/"
    }

    for company in websites:
        print(f"\n{'=' * 45}")
        print(f"АНАЛИЗ КОНКУРЕНТА: {company}")
        print("=" * 45)

        # получаем данные через RAG
        knowledge = search_knowledge(company)
        print(f"\n[RAG] Информация о {company}:")
        print(knowledge)

        # Получение данных через MCP
        print(f"\n[MCP] Запрашиваем сайт {company}...")

        try:
            website_content = asyncio.run(
                fetch_website(websites[company])
            )

            website_content = website_content[:10000]
            print(f"[MCP] Передано символов: {len(website_content)}")
            print("[MCP] Данные сайта получены")

        except Exception as error:
            website_content = ""
            print(f"[MCP] Ошибка: {error}")

        # Анализ конкурента
        prompt = f"""
/no_think
Ты аналитик конкурентной среды.

Проанализируй продукт: {company}

Информация из базы знаний:
{knowledge}

Информация с официального сайта:
{website_content}

Подготовь анализ по структуре:
1. Основные функции (3-5).
2. Конкурентные преимущества (2-3).
3. Целевая аудитория.
4. Возможные ограничения продукта.

Используй только предоставленные сведения.
Если информация не подтверждена, напиши
"Недостаточно данных".

Отвечай кратко и структурированно,
на русском языке.
"""

        response = model.invoke(prompt)
        analysis = response.content

        print("\nРЕЗУЛЬТАТ АНАЛИЗА:")
        print(analysis)

        all_analyses.append(
            f"{company}:\n{analysis}"
        )

    return {
        "all_analyses": "\n\n".join(all_analyses)
    }




# агент 2: сравнительный и стратегический анализ
def strategy_agent(state: AgentState):
    prompt = f"""
/no_think
Ты профессиональный аналитик программных продуктов.

Твоя задача — провести сравнительный анализ
двух конкурентов и предложить обоснованные
рекомендации для развития FlowSpace.

ИНФОРМАЦИЯ О FLOWSPACE:
FlowSpace — платформа для управления задачами
и проектами. Возможности:
- создание и назначение задач;
- Kanban-доски;
- контроль сроков;
- напоминания;
- автоматизация процессов;
- управление проектами.

РЕЗУЛЬТАТЫ АНАЛИЗА КОНКУРЕНТОВ:
{state['all_analyses']}

Сформируй отчёт по структуре:

1. СРАВНЕНИЕ КОНКУРЕНТОВ
Сравни Битрикс24 и ПланФикс по критериям:
- управление задачами и проектами;
- CRM и работа с клиентами;
- автоматизация процессов;
- коммуникации;
- гибкость настройки.

Представь сравнение в виде таблицы.
Если данных недостаточно, укажи
"Недостаточно данных".

2. КОНКУРЕНТНЫЕ ПРЕИМУЩЕСТВА
Укажи по 2 подтверждённых преимущества
каждого продукта.

3. РЕКОМЕНДАЦИИ ДЛЯ FLOWSPACE
Предложи 3 конкретные рекомендации.

Для каждой рекомендации укажи:
- какую функцию стоит добавить или улучшить;
- какая особенность конкурента послужила основанием;
- какую пользу это принесёт FlowSpace.

ВАЖНЫЕ ПРАВИЛА:
- Используй только предоставленные сведения.
- Не придумывай функции и характеристики.
- Не утверждай, что функция отсутствует,
  если это не подтверждено.
- Не противоречь результатам первого агента.
- Не предлагай ограничивать количество
  пользователей без обоснования.
- Пиши на русском языке, структурированно.
"""

    response = model.invoke(prompt)

    return {"recommendations": response.content}



# создание графа
graph = StateGraph(AgentState)

graph.add_node("competitor_agent", competitor_agent)
graph.add_node("strategy_agent", strategy_agent)

graph.add_edge(START, "competitor_agent")
graph.add_edge("competitor_agent", "strategy_agent")
graph.add_edge("strategy_agent", END)

memory = InMemorySaver()
app = graph.compile(checkpointer=memory)



# запуск полного графа LangGraph
config = {
    "configurable": {
        "thread_id": "flowspace_1"
    }
}

result = app.invoke(
    {
        "all_analyses": "",
        "recommendations": ""
    },
    config=config
)

print(f"\n{'=' * 45}")
print("ОБЩЕЕ СРАВНЕНИЕ И РЕКОМЕНДАЦИИ")
print("=" * 45)
print(result["recommendations"])

# проверяем память
saved_state = app.get_state(config)

print("\nПРОВЕРКА ПАМЯТИ:")
print(
    "Анализы двух конкурентов сохранены:",
    all(
        company in saved_state.values["all_analyses"]
        for company in ["Битрикс24", "ПланФикс"]
    )
)
print(
    "Рекомендации сохранены:",
    bool(saved_state.values["recommendations"])
)

# память LangGraph

print("\n" + "=" * 45)
print("ПОВТОРНОЕ ОБРАЩЕНИЕ С ИСПОЛЬЗОВАНИЕМ ПАМЯТИ")
print("=" * 45)

# получаем результаты предыдущего выполнения из памяти
saved_state = app.get_state(config)

previous_analyses = saved_state.values["all_analyses"]

print("\nРезультаты предыдущего анализа загружены из памяти.")

# Новый вопрос на основе сохранённой информации
question = (
    "Какое главное преимущество ПланФикса "
    "стоит использовать для развития FlowSpace?"
)

prompt = f"""
/no_think
Ты стратегический аналитик FlowSpace.

Используй результаты предыдущего анализа:

{previous_analyses}

Ответь на вопрос:
{question}

Предложи одну конкретную рекомендацию.
Не придумывай факты.
Отвечай кратко на русском языке.
"""

response = model.invoke(prompt)

print("\nНОВЫЙ ВОПРОС:")
print(question)

print("\nОТВЕТ НА ОСНОВЕ ПАМЯТИ:")
print(response.content)