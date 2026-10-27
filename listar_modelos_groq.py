import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

chave = os.environ.get("GROQ_API_KEY")

if not chave:
    raise SystemExit("GROQ_API_KEY não encontrada no .env")

client = OpenAI(
    api_key=chave,
    base_url="https://api.groq.com/openai/v1",
)

print("Modelos disponíveis para a sua chave:\n")

for modelo in sorted(client.models.list().data, key=lambda m: m.id):
    print(" -", modelo.id)
