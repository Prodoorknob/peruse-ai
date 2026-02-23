import asyncio
import base64
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage

async def main():
    print("Testing ChatOllama with fake image...")
    # create a tiny fake image
    fake_img = b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06\x00\x00\x00\x1f\x15\xc4\x89\x00\x00\x00\nIDAT\x08\xd7c\xf8\x0f\x00\x01\x01\x01\x00J\xaa\r\x12\x00\x00\x00\x00IEND\xaeB`\x82"
    img_b64 = base64.b64encode(fake_img).decode("utf-8")
    
    llm = ChatOllama(model="qwen3-vl:8b", base_url="http://localhost:11434")
    
    content_blocks = [
        {"type": "text", "text": "What do you see?"},
        {
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{img_b64}"}
        },
        {
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{img_b64}"}
        }
    ]
    
    try:
        response = await llm.ainvoke([HumanMessage(content=content_blocks)])
        print(f"Response: {repr(response.content)}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    asyncio.run(main())
