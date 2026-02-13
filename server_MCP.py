import httpx
import json
import asyncio
from mcp.server.fastmcp import FastMCP
from deep_translator import GoogleTranslator
import typing 



USER_AGENT = "weather-app/1.0"


mcp = FastMCP("weather")

async def make_requests(rus_city: str) -> str:
    headers = {"User-Agent": USER_AGENT, "Accept": "application/json"}
    async with httpx.AsyncClient() as client:
        try:
            city = GoogleTranslator(source='ru', target='en').translate(rus_city)
            url = f"https://api.weatherapi.com/v1/current.json?key=ed0d5e398693402ea3f94009260702&q={city}&aqi=no"
            response = await client.get(url, headers=headers, timeout=30.0)
            data = response.json()

            return f"""
            Location: {data.get("location", {}).get("name", "Unknown")}
            Weather:  {data.get("current", {}).get("condition", {}).get("text", "Unknown")}
            Temp:     {data.get("current", {}).get("temp_c", "Unknown")} °C
            """
        except Exception as e:
            return f"Произошла ошибка: {e}"


@mcp.tool()
async def get_weather(city: str) -> str:
    """Get current weather for a city.

    Args:
        city: Name of the city on English
    """
    result = await make_requests(city)
    return result or "Unable to fetch weather."


def main():
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()