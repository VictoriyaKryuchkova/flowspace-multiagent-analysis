
import asyncio
import sys

from langchain_mcp_adapters.client import MultiServerMCPClient


async def main():
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

    print("MCP-СЕРВЕР ПОДКЛЮЧЁН")
    print("ДОСТУПНЫЕ ИНСТРУМЕНТЫ:")

    for tool in tools:
        print("-", tool.name)

    # Находим инструмент MCP
    fetch_tool = next(
        tool for tool in tools
        if tool.name == "fetch"
    )

    print("\nПРОВЕРКА ВЫЗОВА MCP:")

    # Запрашиваем содержимое тестового сайта
    result = await fetch_tool.ainvoke({
        "url": "https://example.com"
    })

    print(str(result)[:1500])


asyncio.run(main())
