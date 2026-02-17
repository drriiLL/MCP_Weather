import ollama
import asyncio
from aiogram import Dispatcher, Bot
from aiogram.filters import CommandStart, Command
from aiogram.types import Message
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from config import TOKEN
from aiogram.types import ReplyKeyboardRemove

SERVER_PATH = "c:\\VScode\\Python\\Ai\\code\\server_MCP.py"
MODEL = "llama3.2"
SYSTEM_PROMPT = """Ты полезный ассистент.
Вызывай get_weather ТОЛЬКО если пользователь спрашивает о погоде.
Отвечай всегда на русском."""

bot = Bot(token=TOKEN)
dp = Dispatcher()

session = None
tools = []
messages = {}   


async def ask_llm(msgs, tools):
    return ollama.chat(messages=msgs, tools=tools, model=MODEL)


async def call_tool(tool_name, tool_args):
    print(f"\n[Вызываю инструмент: {tool_name} → {tool_args}]")
    result = await session.call_tool(tool_name, tool_args)
    return str(result.content)


async def process_query(query, user_id):
    if user_id not in messages:
        messages[user_id] = [{"role": "system", "content": SYSTEM_PROMPT}]

    messages[user_id].append({"role": "user", "content": query})

    response = await ask_llm(messages[user_id], tools)

    if not response.message.tool_calls:
        answer = response.message.content
        messages[user_id].append({"role": "assistant", "content": answer})
        return answer

    messages[user_id].append({"role": "assistant", "content": ""})

    for tool_call in response.message.tool_calls:
        tool_name = tool_call.function.name
        tool_args = tool_call.function.arguments
        tool_result = await call_tool(tool_name, tool_args)
        messages[user_id].append({"role": "tool", "content": tool_result})

    final = await ask_llm(messages[user_id], tools)
    answer = final.message.content
    messages[user_id].append({"role": "assistant", "content": answer})
    return answer

@dp.message(Command("tools"))
async def start(message: Message):
    await message.answer("Доступные инструменты: ")
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
    for dict in tools:
        slovar = f"name: {dict['function']['name']} \ndescription: {dict['function']['description']}"
        await message.answer(slovar)


@dp.message(CommandStart())
async def start(message: Message):
    
    await message.answer(
        "Привет! Я готов помочь",
        reply_markup=ReplyKeyboardRemove() 
    )
    await message.answer("Доступные инструменты: ")  
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
    for dict in tools:
        slovar = f"name: {dict['function']['name']} \ndescription: {dict['function']['description']}"
        await message.answer(slovar)



@dp.message()
async def handle_message(message: Message):
    user_id = message.from_user.id
    query = message.text

    await message.answer("Думаю...")  

    answer = await process_query(query, user_id)
    await message.answer(answer)


async def main():
    global session, tools

    server_params = StdioServerParameters(
        command="python",
        args=[SERVER_PATH]
    )

    async with stdio_client(server_params) as (stdio, write):
        async with ClientSession(stdio, write) as session:
            await session.initialize()

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

            print(f"MCP подключён. Инструменты: {[t['function']['name'] for t in tools]}")
            print("Бот запущен!")

            
            await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())