import ollama
import asyncio

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


SERVER_PATH = "c:\\VScode\\Python\\Ai\\code\\server_MCP.py"
MODEL = "llama3.2"
SYSTEM_PROMPT = """Ты полезный ассистент.
У тебя есть инструмент get_weather для получения погоды.
Вызывай get_weather ТОЛЬКО если пользователь явно спрашивает о погоде или температуре в каком-то городе.
Если вопрос не связан с погодой — отвечай сам, без вызова инструментов всегда на русском"""



async def ask_llm(message, tools):
    """
    Отправляет и Возвращает ответ от Ollama
    """
    return ollama.chat(messages=message, tools=tools, model=MODEL)

async def process_query(query, tools, session, messages):
    """Обрабатывает запрос пользователя"""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": query}
    ]
    response = await ask_llm(messages, tools)
    if not response.message.tool_calls:
        return response.message.content
    
    for tool_call in response.message.tool_calls:
        tool_name = tool_call.function.name
        tool_args = tool_call.function.arguments

        # Вызываем инструмент на сервере
        tool_result = await call_tool(session, tool_name, tool_args)

        # Добавляем результат в историю
        messages.append({"role": "assistant", "content": str(response.message.content or "")})
        messages.append({"role": "tool", "content": tool_result})
    final = await ask_llm(messages, tools)
    return final.message.content


async def call_tool(session, tool_name, tool_args):
    print(f"\n[Вызываю инструмент: {tool_name} → {tool_args}]")
    result = await session.call_tool(tool_name, tool_args)
    return str(result.content)

async def main():
    
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    
    server_params = StdioServerParameters(
        command="python",
        args=[SERVER_PATH]
    )
    
    async with stdio_client(server_params) as (stdio, write):
        async with ClientSession(stdio, write) as session:
            await session.initialize()

            response = await session.list_tools()
            #print(response)
            
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

            while True:
                query = input("Введите вопрос: ")
                if not query:
                    continue
                if query.lower() == "exit":
                    break

                response = await process_query(query, tools, session, messages)
                print(response)
        

if __name__ == "__main__":
    asyncio.run(main())