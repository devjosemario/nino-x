import os
import json
import queue
import sys
import subprocess
import webbrowser
from datetime import datetime

import sounddevice as sd
from vosk import Model, KaldiRecognizer
import pyttsx3
from openai import OpenAI
from dotenv import load_dotenv


# =========================
# CONFIGURAÇÕES
# =========================

MODELO = "vosk-model-small-pt-0.3"
MICROFONE = 1
TAXA = 16000
MODELO_GROQ = "llama-3.3-70b-versatile"

load_dotenv()

client = OpenAI(
    api_key=os.environ.get("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1",
)


# =========================
# VOZ
# =========================

def falar(texto):
    print(f"NINO: {texto}")

    voz = pyttsx3.init()
    voz.setProperty("rate", 170)
    voz.setProperty("volume", 1.0)
    voz.say(texto)
    voz.runAndWait()
    voz.stop()
    del voz


# =========================
# RECONHECIMENTO (offline, Vosk)
# =========================

print("Carregando modelo de português...")

modelo = Model(MODELO)

fila_audio = queue.Queue()


def callback(indata, frames, tempo, status):
    if status:
        print(status, file=sys.stderr)
    fila_audio.put(bytes(indata))


def ouvir():
    reconhecedor = KaldiRecognizer(modelo, TAXA)

    while True:
        print("\n🎙️ Estou ouvindo...")

        with sd.RawInputStream(
            samplerate=TAXA,
            blocksize=8000,
            device=MICROFONE,
            dtype="int16",
            channels=1,
            callback=callback
        ):
            while True:
                dados = fila_audio.get()

                if reconhecedor.AcceptWaveform(dados):
                    resultado = json.loads(reconhecedor.Result())
                    texto = resultado.get("text", "")

                    if texto:
                        print(f"Você: {texto}")
                        return texto.lower()


# =========================
# AÇÕES REAIS DO SISTEMA
# =========================

def acao_hora():
    return f"São {datetime.now().strftime('%H:%M')}."


def acao_data():
    return f"Hoje é {datetime.now().strftime('%d/%m/%Y')}."


def acao_abrir_calculadora():
    subprocess.Popen("calc.exe")
    return "Abrindo a calculadora."


def acao_abrir_bloco_notas():
    subprocess.Popen("notepad.exe")
    return "Abrindo o bloco de notas."


def acao_abrir_arquivos():
    subprocess.Popen("explorer.exe")
    return "Abrindo o explorador de arquivos."


SITES = {
    "google": ("https://www.google.com", "Google"),
    "youtube": ("https://www.youtube.com", "YouTube"),
    "github": ("https://github.com", "GitHub"),
}


def acao_abrir_site(site):
    url, nome = SITES.get(site, (None, None))
    if not url:
        return "Não conheço esse site."
    webbrowser.open(url)
    return f"Abrindo o {nome}."


def acao_pesquisar(termo):
    url = "https://www.google.com/search?q=" + termo.replace(" ", "+")
    webbrowser.open(url)
    return f"Pesquisando por {termo}."


# =========================
# FERRAMENTAS (formato Chat Completions)
# =========================

FERRAMENTAS = [
    {
        "type": "function",
        "function": {
            "name": "dizer_hora",
            "description": "Informa a hora atual",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "dizer_data",
            "description": "Informa a data atual",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "abrir_calculadora",
            "description": "Abre a calculadora do Windows",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "abrir_bloco_notas",
            "description": "Abre o Bloco de Notas do Windows",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "abrir_arquivos",
            "description": "Abre o explorador de arquivos do Windows",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "abrir_site",
            "description": "Abre um site conhecido no navegador padrão",
            "parameters": {
                "type": "object",
                "properties": {
                    "site": {
                        "type": "string",
                        "enum": list(SITES.keys()),
                        "description": "Qual site abrir",
                    }
                },
                "required": ["site"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "pesquisar",
            "description": "Faz uma pesquisa no Google sobre um termo ou assunto",
            "parameters": {
                "type": "object",
                "properties": {
                    "termo": {"type": "string", "description": "O que pesquisar"}
                },
                "required": ["termo"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "encerrar",
            "description": "Encerra o assistente NINO quando o usuário pede para sair, parar ou desligar",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
]


def executar_ferramenta(nome, argumentos):
    """Executa a ação escolhida pela IA. Retorna (texto_para_falar, continuar_rodando)."""

    if nome == "dizer_hora":
        return acao_hora(), True

    if nome == "dizer_data":
        return acao_data(), True

    if nome == "abrir_calculadora":
        return acao_abrir_calculadora(), True

    if nome == "abrir_bloco_notas":
        return acao_abrir_bloco_notas(), True

    if nome == "abrir_arquivos":
        return acao_abrir_arquivos(), True

    if nome == "abrir_site":
        return acao_abrir_site(argumentos.get("site", "")), True

    if nome == "pesquisar":
        return acao_pesquisar(argumentos.get("termo", "")), True

    if nome == "encerrar":
        return "Até logo. Encerrando o sistema.", False

    return "Não sei executar essa ação.", True


# =========================
# INTERPRETAÇÃO COM IA (Groq)
# =========================

INSTRUCOES_SISTEMA = """
Você é NINO, um assistente virtual pessoal que responde em português do Brasil.
Quando o pedido do usuário corresponder a uma ação conhecida (abrir programas,
abrir sites, pesquisar algo, dizer a hora ou a data, ou encerrar o sistema),
use a ferramenta apropriada.
Se for uma pergunta geral, um pedido de conversa ou algo sem ferramenta
correspondente, apenas responda em texto, de forma breve, educada e objetiva.
Não diga que é o NINO original do Homem de Ferro.
"""


def processar_comando(comando):
    resposta = client.chat.completions.create(
        model=MODELO_GROQ,
        messages=[
            {"role": "system", "content": INSTRUCOES_SISTEMA},
            {"role": "user", "content": comando},
        ],
        tools=FERRAMENTAS,
    )

    mensagem = resposta.choices[0].message
    continuar = True

    if mensagem.tool_calls:
        for chamada in mensagem.tool_calls:
            nome = chamada.function.name
            argumentos = json.loads(chamada.function.arguments) if chamada.function.arguments else {}
            texto_resposta, continuar = executar_ferramenta(nome, argumentos)
            falar(texto_resposta)
    else:
        falar(mensagem.content)

    return continuar


# =========================
# INICIALIZAÇÃO
# =========================

falar("Sistemas iniciados. Olá. Eu sou NINO. Estou pronto para receber seus comandos.")


# =========================
# LOOP PRINCIPAL
# =========================

while True:
    comando = ouvir()
    continuar = processar_comando(comando)

    if not continuar:
        break