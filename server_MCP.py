import httpx
import json
import asyncio
from mcp.server.fastmcp import FastMCP
from deep_translator import GoogleTranslator
import typing 
from docx import Document
import os

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

def get_doc_text(name: str) -> list:
    doc = Document(os.path.join("C:\\VScode\\Python\\Ai\\code", name))
    text = []
    for paragraph in doc.paragraphs:
        if paragraph.text.strip():
            text.append(paragraph.text)
    for table in doc.tables:
        for row in table.rows:
            for cell in row.cells:
                if cell.text.strip():
                    text.append(cell.text)
    return text

@mcp.tool()
async def create_doc(name: str, text: str) -> str:
    """
    Tool for create_doc
    
    Args:
        name: Name of file
        text: Text which need paste in document
    """

    doc = Document()
    try:
        doc.add_paragraph(text)
        doc.save(name)
        return "Готово"
    except Exception as e:
        return f"Произошла ошибка {e}"



@mcp.tool()
async def get_weather(city: str) -> str:
    """Get current weather for a city.

    Args:
        city: Name of the city on English
    """
    result = await make_requests(city)
    return result or "Unable to fetch weather."



@mcp.tool()
async def read_doc(name: str) -> str:
    """Read word document and return its text content.
    
    Args:
        name: Name of word file with extension, e.g. 'document.docx'
    """
    result = get_doc_text(name)
    return "\n".join(result)


def main():
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()