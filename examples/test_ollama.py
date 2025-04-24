# test_ollama.py
import os

from dotenv import load_dotenv
from langchain_ollama import ChatOllama

# 1) Load your .env.development automatically
load_dotenv(".env.development")

# 2) Create the client
ollama_url = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434")
model_name = os.getenv("LLM_MODEL", "llama3.2:3b")
temperature = float(os.getenv("DEFAULT_LLM_TEMPERATURE", "0.2"))

client = ChatOllama(
    base_url=ollama_url,
    model=model_name,
    temperature=temperature
)

# 3) Send a quick prompt
response = client.invoke([{"role": "user", "content": "Hello Ollama, how are you today?"}])
# print("Response type:", type(response))
# print("Response structure:", dir(response))
# print("Response content:", response.content)
print("Ollama says:", response.content)