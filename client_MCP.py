import asyncio
import ollama
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

SERVER_PATH = "c:\\VScode\\Python\\Ai\\code\\server_MCP.py"
MODEL = "llama3.2"

SYSTEM_PROMPT = """Ты полезный ассистент.
У тебя есть инструмент get_weather для получения погоды.
Вызывай get_weather ТОЛЬКО если пользователь явно спрашивает о погоде или температуре в каком-то городе.
Если вопрос не связан с погодой — отвечай сам, без вызова инструментов всегда на русском"""


async def call_tool(session: ClientSession, tool_name: str, tool_args: dict) -> str:
    """Вызывает инструмент на MCP сервере и возвращает результат."""
    print(f"\n[Вызываю инструмент: {tool_name} → {tool_args}]")
    result = await session.call_tool(tool_name, tool_args)
    return str(result.content)


async def ask_llm(messages: list, tools: list) -> object:
    """Отправляет сообщения в Ollama и возвращает ответ."""
    return ollama.chat(
        model=MODEL,
        messages=messages,
        tools=tools
    )


async def process_query(query: str, session: ClientSession, tools: list) -> str:
    """Обрабатывает запрос пользователя."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": query}
    ]

    # Первый запрос в Llama
    response = await ask_llm(messages, tools)

    # Если Llama не хочет вызывать инструмент — возвращаем ответ сразу
    if not response.message.tool_calls:
        return response.message.content

    # Llama хочет вызвать инструмент
    for tool_call in response.message.tool_calls:
        tool_name = tool_call.function.name
        tool_args = tool_call.function.arguments

        # Вызываем инструмент на сервере
        tool_result = await call_tool(session, tool_name, tool_args)

        # Добавляем результат в историю
        messages.append({"role": "assistant", "content": str(response.message.content or "")})
        messages.append({"role": "tool", "content": tool_result})

    # Финальный запрос — Llama формулирует ответ на основе результата инструмента
    final_response = await ask_llm(messages, tools)
    return final_response.message.content


async def main():
    server_params = StdioServerParameters(
        command="python",
        args=[SERVER_PATH]
    )

    async with stdio_client(server_params) as (stdio, write):
        async with ClientSession(stdio, write) as session:
            await session.initialize()

            # Получаем список инструментов с сервера
            response = await session.list_tools()
            tools = [
                {
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": tool.inputSchema
                    }
                }
                for tool in response.tools
            ]

            print(f"Подключено. Доступные инструменты: {[t['function']['name'] for t in tools]}")
            print("Введите 'выход' для завершения\n")

            while True:
                query = input("Вопрос: ").strip()

                if not query:
                    continue

                if query.lower() == "выход":
                    break

                try:
                    answer = await process_query(query, session, tools)
                    print(f"\nОтвет: {answer}\n")
                except Exception as e:
                    print(f"\nОшибка: {e}\n")


if __name__ == "__main__":
    asyncio.run(main())