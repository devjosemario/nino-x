import os
import json
import queue
import sys
import subprocess
import webbrowser
import asyncio
import uuid

import edge_tts
from playsound import playsound
from datetime import datetime

import numpy as np
import sounddevice as sd
from vosk import Model, KaldiRecognizer
from openai import OpenAI
from dotenv import load_dotenv

import openwakeword
from openwakeword.model import Model as ModeloWakeword


# ============================================================
# CONFIGURAÇÕES
# ============================================================

MODELO = "vosk-model-small-pt-0.3"

# Microfone L80
MICROFONE = 1

# Taxa de amostragem
TAXA = 16000

# Modelo da IA Groq
MODELO_GROQ = "llama-3.3-70b-versatile"


# ============================================================
# CONFIGURAÇÃO DA API GROQ
# ============================================================

load_dotenv()

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

if not GROQ_API_KEY:
    print("ERRO: GROQ_API_KEY não encontrada no arquivo .env")
    sys.exit(1)


client = OpenAI(
    api_key=GROQ_API_KEY,
    base_url="https://api.groq.com/openai/v1",
)


# ============================================================
# CONFIGURAÇÃO DA VOZ
# ============================================================

VOZ_EDGE = "pt-BR-AntonioNeural"

# Velocidade da voz
VELOCIDADE_EDGE = "+10%"


async def gerar_audio(texto, arquivo):

    comunicador = edge_tts.Communicate(
        texto,
        VOZ_EDGE,
        rate=VELOCIDADE_EDGE
    )

    await comunicador.save(arquivo)


def falar(texto):

    if not texto:
        return

    print(f"NINO: {texto}")

    arquivo = f"NINO_{uuid.uuid4().hex}.mp3"

    try:

        # Gera o áudio
        asyncio.run(
            gerar_audio(texto, arquivo)
        )

        # Reproduz o áudio
        playsound(arquivo)

    except Exception as erro:

        print(f"Erro no áudio: {erro}")

    finally:

        # Remove o MP3 temporário
        if os.path.exists(arquivo):

            try:
                os.remove(arquivo)

            except PermissionError:
                pass


# ============================================================
# CARREGAR MODELO VOSK
# ============================================================

print("Carregando modelo de português...")

try:

    modelo = Model(MODELO)

except Exception as erro:

    print("Erro ao carregar o modelo Vosk.")
    print(erro)

    sys.exit(1)


# ============================================================
# FILA DE ÁUDIO
# ============================================================

fila_audio = queue.Queue()


def callback(indata, frames, tempo, status):

    if status:

        print(
            status,
            file=sys.stderr
        )

    fila_audio.put(
        bytes(indata)
    )


# ============================================================
# OUVIR MICROFONE
# ============================================================

def ouvir():

    reconhecedor = KaldiRecognizer(
        modelo,
        TAXA
    )

    print("\n🎙️ Estou ouvindo...")

    try:

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

                    resultado = json.loads(
                        reconhecedor.Result()
                    )

                    texto = resultado.get(
                        "text",
                        ""
                    )

                    if texto:

                        print(
                            f"Você: {texto}"
                        )

                        return texto.lower()

    except Exception as erro:

        print(
            f"Erro no microfone: {erro}"
        )

        return ""


# ============================================================
# DETECTAR PALAVRA DE ATIVAÇÃO
# ============================================================

def detectar_NINO(texto):

    """
    Verifica se a palavra de ativação foi detectada no comando.

    Obs: testes mostraram que o modelo Vosk (small-pt-0.3) converte
    consistentemente "NINO" falado para "jardim" -- por isso a lista
    abaixo aceita as variações que o reconhecedor realmente produz,
    não só a grafia literal "NINO".
    """

    palavras_de_ativacao = ["NINO", "jardim", "jarves"]

    texto = texto.lower()

    return any(palavra in texto for palavra in palavras_de_ativacao)


# ============================================================
# AÇÕES DO SISTEMA
# ============================================================

def acao_hora():

    return (
        f"São "
        f"{datetime.now().strftime('%H:%M')}."
    )


def acao_data():

    return (
        f"Hoje é "
        f"{datetime.now().strftime('%d/%m/%Y')}."
    )


def acao_abrir_calculadora():

    subprocess.Popen(
        "calc.exe"
    )

    return "Abrindo a calculadora."


def acao_abrir_bloco_notas():

    subprocess.Popen(
        "notepad.exe"
    )

    return "Abrindo o Bloco de Notas."


def acao_abrir_arquivos():

    subprocess.Popen(
        "explorer.exe"
    )

    return "Abrindo o explorador de arquivos."


# ============================================================
# SITES
# ============================================================

SITES = {

    "google": (
        "https://www.google.com",
        "Google"
    ),

    "youtube": (
        "https://www.youtube.com",
        "YouTube"
    ),

    "github": (
        "https://github.com",
        "GitHub"
    ),

}


def acao_abrir_site(site):

    url, nome = SITES.get(
        site,
        (None, None)
    )

    if not url:

        return "Não conheço esse site."

    webbrowser.open(url)

    return f"Abrindo o {nome}."


# ============================================================
# PESQUISA NO GOOGLE
# ============================================================

def acao_pesquisar(termo):

    url = (
        "https://www.google.com/search?q="
        + termo.replace(" ", "+")
    )

    webbrowser.open(url)

    return f"Pesquisando por {termo}."


# ============================================================
# FERRAMENTAS DA IA
# ============================================================

FERRAMENTAS = [

    {
        "type": "function",

        "function": {

            "name": "dizer_hora",

            "description":
                "Informa a hora atual.",

            "parameters": {

                "type": "object",

                "properties": {},

                "required": []

            },

        },

    },

    {
        "type": "function",

        "function": {

            "name": "dizer_data",

            "description":
                "Informa a data atual.",

            "parameters": {

                "type": "object",

                "properties": {},

                "required": []

            },

        },

    },

    {
        "type": "function",

        "function": {

            "name": "abrir_calculadora",

            "description":
                "Abre a calculadora do Windows.",

            "parameters": {

                "type": "object",

                "properties": {},

                "required": []

            },

        },

    },

    {
        "type": "function",

        "function": {

            "name": "abrir_bloco_notas",

            "description":
                "Abre o Bloco de Notas do Windows.",

            "parameters": {

                "type": "object",

                "properties": {},

                "required": []

            },

        },

    },

    {
        "type": "function",

        "function": {

            "name": "abrir_arquivos",

            "description":
                "Abre o explorador de arquivos do Windows.",

            "parameters": {

                "type": "object",

                "properties": {},

                "required": []

            },

        },

    },

    {
        "type": "function",

        "function": {

            "name": "abrir_site",

            "description":
                "Abre um site conhecido no navegador padrão.",

            "parameters": {

                "type": "object",

                "properties": {

                    "site": {

                        "type": "string",

                        "enum":
                            list(SITES.keys()),

                        "description":
                            "Qual site abrir."

                    }

                },

                "required": [
                    "site"
                ]

            },

        },

    },

    {
        "type": "function",

        "function": {

            "name": "pesquisar",

            "description":
                "Faz uma pesquisa no Google sobre um termo ou assunto.",

            "parameters": {

                "type": "object",

                "properties": {

                    "termo": {

                        "type": "string",

                        "description":
                            "O que pesquisar."

                    }

                },

                "required": [
                    "termo"
                ]

            },

        },

    },

    {
        "type": "function",

        "function": {

            "name": "encerrar",

            "description":
                "Encerra o assistente NINO quando o usuário pede para sair, parar ou desligar.",

            "parameters": {

                "type": "object",

                "properties": {},

                "required": []

            },

        },

    },

]


# ============================================================
# EXECUTAR FERRAMENTA
# ============================================================

def executar_ferramenta(nome, argumentos):

    if nome == "dizer_hora":

        return (
            acao_hora(),
            True
        )


    if nome == "dizer_data":

        return (
            acao_data(),
            True
        )


    if nome == "abrir_calculadora":

        return (
            acao_abrir_calculadora(),
            True
        )


    if nome == "abrir_bloco_notas":

        return (
            acao_abrir_bloco_notas(),
            True
        )


    if nome == "abrir_arquivos":

        return (
            acao_abrir_arquivos(),
            True
        )


    if nome == "abrir_site":

        return (
            acao_abrir_site(
                argumentos.get(
                    "site",
                    ""
                )
            ),
            True
        )


    if nome == "pesquisar":

        return (
            acao_pesquisar(
                argumentos.get(
                    "termo",
                    ""
                )
            ),
            True
        )


    if nome == "encerrar":

        return (
            "Até logo. Encerrando o sistema.",
            False
        )


    return (
        "Não sei executar essa ação.",
        True
    )


# ============================================================
# PERSONALIDADE DO NINO
# ============================================================

INSTRUCOES_SISTEMA = (

    "Você é NINO, um assistente virtual pessoal "
    "que responde em português do Brasil. "

    "Sua personalidade é educada, inteligente, "
    "objetiva e prestativa. "

    "Quando o pedido do usuário corresponder "
    "a uma ação conhecida, use a ferramenta apropriada. "

    "As ações disponíveis incluem abrir programas, "
    "abrir sites, pesquisar algo, informar hora, "
    "informar data e encerrar o sistema. "

    "Se for uma pergunta geral, um pedido de conversa "
    "ou algo sem ferramenta correspondente, "
    "responda em texto de forma breve, educada e objetiva. "

    "Não diga que é o NINO original do Homem de Ferro."

)


# ============================================================
# PROCESSAR COMANDO COM GROQ
# ============================================================

def processar_comando(comando):

    if not comando:

        return True


    try:

        resposta = client.chat.completions.create(

            model=MODELO_GROQ,

            messages=[

                {
                    "role": "system",
                    "content": INSTRUCOES_SISTEMA
                },

                {
                    "role": "user",
                    "content": comando
                }

            ],

            tools=FERRAMENTAS,

        )

    except Exception as erro:

        print(
            f"Erro na IA: {erro}"
        )

        falar(
            "Desculpe. "
            "Não consegui acessar meu sistema de inteligência."
        )

        return True


    mensagem = (
        resposta
        .choices[0]
        .message
    )

    continuar = True


    # ========================================================
    # IA ESCOLHE UMA FERRAMENTA
    # ========================================================

    if mensagem.tool_calls:

        for chamada in mensagem.tool_calls:

            nome = (
                chamada
                .function
                .name
            )

            argumentos = (

                json.loads(
                    chamada
                    .function
                    .arguments
                )

                if chamada.function.arguments

                else {}

            )


            texto_resposta, continuar = (
                executar_ferramenta(
                    nome,
                    argumentos
                )
            )


            falar(
                texto_resposta
            )


    # ========================================================
    # IA RESPONDE NORMALMENTE
    # ========================================================

    else:

        texto = (
            mensagem.content
            or
            "Não consegui formular uma resposta."
        )

        falar(texto)


    return continuar


# ============================================================
# INICIALIZAÇÃO
# ============================================================

falar(
    "Sistemas iniciados. "
    "Estou aguardando a palavra de ativação."
)


# ============================================================
# LOOP PRINCIPAL
# ============================================================

while True:

    # ========================================================
    # MODO DE ESPERA
    # ========================================================

    comando = ouvir()

    if not comando:

        continue


    # ========================================================
    # VERIFICAR SE FALOU NINO
    # ========================================================

    if not detectar_NINO(comando):

        continue


    # ========================================================
    # NINO FOI CHAMADO
    # ========================================================

    falar("Sim?")


    # ========================================================
    # AGUARDAR COMANDO
    # ========================================================

    comando = ouvir()

    if not comando:

        continue


    # ========================================================
    # PROCESSAR COMANDO
    # ========================================================

    continuar = processar_comando(
        comando
    )


    # ========================================================
    # ENCERRAR
    # ========================================================

    if not continuar:

        break