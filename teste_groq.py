import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

chave = os.environ.get("GROQ_API_KEY")

print(f"Chave encontrada: {'sim' if chave else 'NAO'}")
if chave:
    print(f"Começa com: {chave[:6]}...")

client = OpenAI(
    api_key=chave,
    base_url="https://api.groq.com/openai/v1",
)

print("Enviando pergunta de teste para o Groq...")

resposta = client.chat.completions.create(
    model="llama-3.3-70b-versatile",
    messages=[
        {"role": "user", "content": "Diga apenas 'funcionando' e nada mais."}
    ],
)

print("Resposta recebida:")
print(resposta.choices[0].message.content)
