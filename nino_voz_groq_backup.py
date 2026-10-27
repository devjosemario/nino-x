import os
import re
import json
import queue
import sys
import subprocess
import webbrowser
import asyncio
import time
import uuid
import unicodedata
import urllib.error
from urllib.parse import quote_plus
import urllib.request
from collections import deque
from datetime import datetime

import edge_tts
import numpy as np
from playsound3 import playsound

import sounddevice as sd
from vosk import Model, KaldiRecognizer
from faster_whisper import WhisperModel
from openai import OpenAI
from dotenv import load_dotenv
from configuracao_nino import carregar_configuracao, definir_modo, mostrar_aviso

load_dotenv()

from controle_tela import (
    abrir_aplicativo_menu_iniciar,
    fechar_janela_local,
    executar_controle_tela,
    interpretar_confirmacao,
)
from memoria_nino import contexto_memoria, processar_comando_memoria


# ============================================================
# CONFIGURAÇÕES
# ============================================================

MODELO = "vosk-model-small-pt-0.3"
MICROFONE = 1                       # índice do seu microfone (veja teste_microfone.py)
TAXA = 16000
MODELO_WHISPER = "small"
LIMIAR_AUDIO = int(os.environ.get("NINO_LIMIAR_AUDIO", "260"))
SILENCIO_FINAL_MS = 700
DURACAO_MAXIMA_FALA = 18
BLOCO_AUDIO = 1600                  # 100 ms em 16 kHz

VOZ_EDGE = "pt-BR-AntonioNeural"
VELOCIDADE_EDGE = "+10%"

# --- Cérebro 1: OpenJarvis (servidor local: `jarvis serve --port 8000`) ---
OPENJARVIS_URL = os.environ.get("OPENJARVIS_URL", "http://localhost:8000/v1")

# --- Cérebro 2: Groq (nuvem, reserva) ---
GROQ_URL = "https://api.groq.com/openai/v1"
MODELO_GROQ = "openai/gpt-oss-120b"
GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
MODELO_GEMINI = os.environ.get("NINO_GEMINI_MODEL", "gemini-3.8-flash")
OLLAMA_URL = os.environ.get("NINO_LOCAL_CHAT_URL", "http://localhost:11434/v1")

# Quantas mensagens recentes o Nino lembra durante a conversa
HISTORICO_MAX = 8


# ============================================================
# ATIVAÇÃO POR VOZ ("Nino" / "oi Nino")
# ============================================================
# O Vosk (small-pt) ouve "Nino" como "nino", "menino", "menina",
# "menu", "o menino", "me no", "nem no"...
# O chamado só vale se estiver no COMEÇO da frase.

SAUDACOES = ["oi", "ei", "ô", "olá"]
APELIDOS_NINO = ["nino", "menino", "menina", "menu", "milho", "mino", "nina"]
PREFIXOS = ["o", "a", "me", "nem", "do"]
GRAMATICA_ATIVACAO = json.dumps(
    [
        "nino", "oi nino", "ei nino", "ola nino", "[unk]",
        "nino voce ta ai", "nino voce esta ai", "nino ta ai",
        "oi nino voce ta ai", "oi nino voce esta ai",
    ],
    ensure_ascii=False,
)


# ============================================================
# ESTADO GLOBAL (preenchido no main)
# ============================================================

modelo = None
modelo_whisper = None
CLIENTES = []          # lista de (nome, grupo, cliente, nome_do_modelo)
historico = []         # conversa recente
fila_audio = queue.Queue()
fila_comandos = queue.Queue()
TAREFA_TELA_PENDENTE = None
TAREFA_TELA_REPETIR = None
MODO_SUSPENSO = False
PROVEDOR_ATUAL = None
ULTIMO_ERRO_IA = None


# ============================================================
# VOZ (edge-tts)
# ============================================================

async def gerar_audio(texto, arquivo):
    comunicador = edge_tts.Communicate(texto, VOZ_EDGE, rate=VELOCIDADE_EDGE)
    await comunicador.save(arquivo)


def falar(texto):
    if not texto:
        return

    print(f"NINO: {texto}")

    arquivo = f"nino_{uuid.uuid4().hex}.mp3"

    try:
        asyncio.run(gerar_audio(texto, arquivo))
        playsound(arquivo)

    except Exception as erro:
        print(f"Erro no áudio: {erro}")

    finally:
        if os.path.exists(arquivo):
            try:
                os.remove(arquivo)
            except OSError:
                pass


# ============================================================
# RECONHECIMENTO DE VOZ (Faster-Whisper ativo; Vosk em espera)
# ============================================================

def callback(indata, frames, tempo, status):
    if status:
        print(status, file=sys.stderr)
    fila_audio.put(bytes(indata))


def limpar_fila():
    while not fila_audio.empty():
        try:
            fila_audio.get_nowait()
        except queue.Empty:
            break


def _ouvir_vosk(stop_event=None, gramatica=None):
    limpar_fila()
    if gramatica:
        reconhecedor = KaldiRecognizer(modelo, TAXA, gramatica)
    else:
        reconhecedor = KaldiRecognizer(modelo, TAXA)

    try:
        with sd.RawInputStream(
            samplerate=TAXA,
            blocksize=BLOCO_AUDIO,
            device=MICROFONE,
            dtype="int16",
            channels=1,
            callback=callback,
        ):
            while True:
                if stop_event and stop_event.is_set():
                    return ""

                try:
                    dados = fila_audio.get(timeout=0.25)
                except queue.Empty:
                    continue

                if reconhecedor.AcceptWaveform(dados):
                    resultado = json.loads(reconhecedor.Result())
                    texto = resultado.get("text", "")

                    if texto:
                        print(f"Você: {texto}")
                        return texto.lower()

    except Exception as erro:
        print(f"Erro no microfone: {erro}")
        return ""


def _transcrever_whisper(audio):
    inicio = time.perf_counter()
    segmentos, _ = modelo_whisper.transcribe(
        audio,
        language="pt",
        beam_size=1,
        vad_filter=True,
        condition_on_previous_text=False,
    )
    texto = " ".join(segmento.text.strip() for segmento in segmentos).strip()
    if texto:
        print(f"Transcrição concluída em {time.perf_counter() - inicio:.2f} s")
        print(f"Você: {texto}")
    return texto.lower()


def _ouvir_whisper(stop_event=None):
    """Grava uma fala por vez e encerra após 700 ms de silêncio."""
    limpar_fila()
    blocos_anteriores = deque(maxlen=3)
    audio_fala = []
    iniciou_fala = False
    silencio_amostras = 0
    total_amostras = 0
    max_amostras = DURACAO_MAXIMA_FALA * TAXA

    try:
        with sd.RawInputStream(
            samplerate=TAXA,
            blocksize=BLOCO_AUDIO,
            device=MICROFONE,
            dtype="int16",
            channels=1,
            callback=callback,
        ):
            while True:
                if stop_event and stop_event.is_set():
                    return ""

                try:
                    dados = fila_audio.get(timeout=0.25)
                except queue.Empty:
                    continue

                amostras = np.frombuffer(dados, dtype=np.int16)
                nivel = float(np.sqrt(np.mean(amostras.astype(np.float32) ** 2)))

                if nivel >= LIMIAR_AUDIO:
                    if not iniciou_fala:
                        audio_fala.extend(blocos_anteriores)
                        total_amostras = sum(len(bloco) // 2 for bloco in audio_fala)
                        iniciou_fala = True
                    audio_fala.append(dados)
                    total_amostras += len(amostras)
                    silencio_amostras = 0
                elif iniciou_fala:
                    audio_fala.append(dados)
                    total_amostras += len(amostras)
                    silencio_amostras += len(amostras)
                    if silencio_amostras >= TAXA * SILENCIO_FINAL_MS // 1000:
                        break
                else:
                    blocos_anteriores.append(dados)

                if total_amostras >= max_amostras:
                    break

    except Exception as erro:
        print(f"Erro no microfone: {erro}")
        return ""

    if not iniciou_fala:
        return ""

    pcm = np.frombuffer(b"".join(audio_fala), dtype=np.int16)
    audio = pcm.astype(np.float32) / 32768.0
    return _transcrever_whisper(audio)


def ouvir(espera_ativacao=False, stop_event=None, gramatica=None):
    if espera_ativacao or gramatica or modelo_whisper is None:
        gramatica_vosk = gramatica or (GRAMATICA_ATIVACAO if espera_ativacao else None)
        return _ouvir_vosk(stop_event, gramatica_vosk)
    return _ouvir_whisper(stop_event)


# ============================================================
# DETECTAR "NINO"
# ============================================================

def _fim_do_chamado(palavras):
    """Procura o chamado no início da frase.
    Retorna a posição onde ele termina, ou -1 se não houver."""

    i = 0

    while (
        i < len(palavras)
        and i < 3
        and (palavras[i] in SAUDACOES or palavras[i] in PREFIXOS)
    ):
        i += 1

    if i < len(palavras) and palavras[i] in APELIDOS_NINO:
        return i + 1

    if (
        i > 0
        and i < len(palavras)
        and palavras[i] == "no"
        and palavras[i - 1] in ("me", "nem")
    ):
        return i + 1

    if palavras == ["no"]:
        return 1

    return -1


def detectar_nino(texto):
    return _fim_do_chamado(texto.lower().split()) != -1


def extrair_comando(texto):
    """'menino que horas são' -> 'que horas são'."""
    palavras = texto.lower().split()
    fim = _fim_do_chamado(palavras)

    if fim == -1:
        return texto.lower()

    return " ".join(palavras[fim:])


def normalizar_texto(texto):
    texto = unicodedata.normalize("NFKD", texto.lower())
    texto = "".join(caractere for caractere in texto if not unicodedata.combining(caractere))
    texto = re.sub(r"[^\w\s]", " ", texto)
    return re.sub(r"\s+", " ", texto).strip()


def comando_para_suspender(texto):
    frase = normalizar_texto(texto)
    comandos = {
        "desligar", "desliga", "desligue", "nino desligar", "nino desliga",
        "desligar nino", "desliga nino", "nino desligue", "desligue nino",
        "pode parar de falar", "nino pode parar de falar", "pode parar nino",
        "para de falar", "pare de falar", "nino para de falar",
        "nao preciso mais de voce", "nao preciso mais", "nino nao preciso mais",
    }
    return frase in comandos


def pergunta_se_nino_esta_ativo(texto):
    frase = normalizar_texto(texto)
    return frase in {"voce ta ai", "voce esta ai", "ta ai", "esta ai"}


# ============================================================
# AÇÕES DO SISTEMA
# ============================================================

def acao_hora():
    return f"São {datetime.now().strftime('%H:%M')}."


def acao_data():
    return f"Hoje é {datetime.now().strftime('%d/%m/%Y')}."


def acao_abrir_calculadora():
    subprocess.Popen("calc.exe")
    return "Abrindo a calculadora."


def acao_abrir_bloco_notas():
    subprocess.Popen("notepad.exe")
    return "Abrindo o Bloco de Notas."


def acao_abrir_arquivos():
    subprocess.Popen("explorer.exe")
    return "Abrindo o explorador de arquivos."


APLICATIVOS = {
    "paint": ("mspaint.exe", "Paint"),
    "gerenciador de tarefas": ("taskmgr.exe", "Gerenciador de Tarefas"),
    "configurações": ("ms-settings:", "Configurações do Windows"),
}


def acao_abrir_aplicativo(aplicativo):
    comando, nome = APLICATIVOS.get(aplicativo, (None, None))

    if not comando:
        return abrir_aplicativo_menu_iniciar(aplicativo)

    try:
        if comando.startswith("ms-settings:"):
            os.startfile(comando)
        else:
            subprocess.Popen([comando])
    except OSError:
        return f"Não consegui abrir {nome}."

    return f"Abrindo {nome}."


PASTAS = {
    "downloads": ("Downloads", "Downloads"),
    "documentos": ("Documents", "Documentos"),
    "área de trabalho": ("Desktop", "área de trabalho"),
}


def acao_abrir_pasta(pasta):
    pasta_sistema, nome = PASTAS.get(pasta, (None, None))

    if not pasta_sistema:
        return "Essa pasta não está na minha lista de pastas permitidas."

    caminho = os.path.join(os.path.expanduser("~"), pasta_sistema)

    try:
        subprocess.Popen(["explorer.exe", caminho])
    except OSError:
        return f"Não consegui abrir a pasta {nome}."

    return f"Abrindo a pasta {nome}."


SITES = {
    "google": ("https://www.google.com", "Google"),
    "youtube": ("https://www.youtube.com", "YouTube"),
    "whatsapp": ("https://web.whatsapp.com", "WhatsApp Web"),
    "github": ("https://github.com", "GitHub"),
}


def acao_abrir_site(site):
    url, nome = SITES.get(site, (None, None))

    if not url:
        return "Não conheço esse site."

    webbrowser.open(url)
    return f"Abrindo o {nome}."


def pedido_abrir_whatsapp(comando):
    frase = normalizar_texto(comando)
    return "whatsapp" in frase and bool(
        re.search(r"\b(abrir|abra|abre|acessar|acesse|acessa|entrar|entre)\b", frase)
    )


def pedido_fechar_janela(comando):
    """Extrai o alvo de um pedido direto para fechar app, janela ou aba."""
    frase = normalizar_texto(comando)
    correspondencia = re.search(
        r"\b(?:feche|fecha|fechar|encerre|encerra|encerrar)\b\s+(.+)",
        frase,
    )
    if not correspondencia:
        return None

    alvo = correspondencia.group(1)
    alvo = re.split(r"\b(?:e depois|depois|por favor)\b", alvo, maxsplit=1)[0]
    fechar_aba = bool(re.search(r"\baba\b", alvo))
    alvo = re.sub(r"^(?:(?:a|o|essa|esta|esse|este)\s+)?aba\s+(?:(?:do|da|de)\s+)?", "", alvo)
    alvo = re.sub(r"^(?:(?:o|a|os|as|meu|minha)\s+)?(?:aplicativo|programa|janela|app)\s+(?:(?:do|da|de)\s+)?", "", alvo)
    alvo = re.sub(r"\s+(?:aberto|aberta|abertos|abertas)$", "", alvo).strip()
    if not alvo or alvo in {"essa", "esta", "isso", "ele", "ela"}:
        return None
    return alvo, fechar_aba


def acao_pesquisar(termo):
    url = "https://www.google.com/search?q=" + termo.replace(" ", "+")
    webbrowser.open(url)
    return f"Pesquisando por {termo}."


def pesquisa_youtube_pedida(comando):
    """Extrai uma busca do YouTube para abrir diretamente os resultados."""
    frase = normalizar_texto(comando)
    if "youtube" not in frase:
        return None

    busca = re.search(
        r"\b(?:pesquise|pesquisa|pesquisar|busque|busca|buscar|procure|procura|procurar)\b\s+(.+)",
        frase,
    )
    if not busca:
        return None

    termo = busca.group(1).strip()
    termo = re.sub(r"^(?:rapido|rapidamente|por favor)\s+", "", termo)
    termo = re.sub(r"\s+(?:no|pelo) youtube$", "", termo).strip()
    return termo or None


def pesquisar_youtube(termo):
    url = "https://www.youtube.com/results?search_query=" + quote_plus(termo)
    webbrowser.open(url)
    return f"Certo, pesquisando {termo} no YouTube."


# ============================================================
# FERRAMENTAS DA IA
# ============================================================

def _ferramenta(nome, descricao, propriedades=None, obrigatorios=None):
    return {
        "type": "function",
        "function": {
            "name": nome,
            "description": descricao,
            "parameters": {
                "type": "object",
                "properties": propriedades or {},
                "required": obrigatorios or [],
            },
        },
    }


FERRAMENTAS = [
    _ferramenta("dizer_hora", "Informa a hora atual."),
    _ferramenta("dizer_data", "Informa a data atual."),
    _ferramenta("abrir_calculadora", "Abre a calculadora do Windows."),
    _ferramenta("abrir_bloco_notas", "Abre o Bloco de Notas do Windows."),
    _ferramenta("abrir_arquivos", "Abre o explorador de arquivos do Windows."),
    _ferramenta(
        "abrir_aplicativo",
        "Abre qualquer aplicativo instalado pelo menu Iniciar do Windows.",
        {"aplicativo": {"type": "string", "description": "Nome do aplicativo instalado"}},
        ["aplicativo"],
    ),
    _ferramenta(
        "abrir_pasta",
        "Abre uma pasta padrão do usuário: Downloads, Documentos ou área de trabalho.",
        {"pasta": {"type": "string", "enum": list(PASTAS), "description": "Pasta padrão"}},
        ["pasta"],
    ),
    _ferramenta(
        "criar_pasta",
        "Cria uma pasta no caminho informado, incluindo pastas intermediárias necessárias.",
        {
            "caminho": {
                "type": "string",
                "description": "Caminho completo ou relativo da nova pasta",
            }
        },
        ["caminho"],
    ),
    _ferramenta(
        "criar_mesa_3d",
        "Cria uma mesa 3D no Blender pela ponte local do NinoX. Use quando o usuário pedir um modelo de mesa no Blender. "
        "Todas as medidas são em metros; se não forem informadas, escolha dimensões comuns e explique quais usou.",
        {
            "largura": {"type": "number", "description": "Largura em metros"},
            "profundidade": {"type": "number", "description": "Profundidade em metros"},
            "diametro": {"type": "number", "description": "Diâmetro em metros; use para tampo redondo"},
            "altura": {"type": "number", "description": "Altura total em metros"},
            "espessura": {"type": "number", "description": "Espessura do tampo em metros"},
            "nome": {"type": "string", "description": "Nome do modelo"},
        },
    ),
    _ferramenta(
        "controlar_tela",
        "Use para entender o que aparece na tela ou executar uma tarefa visual no computador. "
        "O usuário já autorizou acesso contínuo à tela para executar as ordens que der. "
        "Execute as ordens claras. Descreva a tarefa em português e pergunte apenas quando o alvo ou a ordem forem ambíguos.",
        {"tarefa": {"type": "string", "description": "O que o usuário quer fazer ou entender na tela"}},
        ["tarefa"],
    ),
    _ferramenta(
        "abrir_site",
        "Abre um site conhecido no navegador padrão.",
        {
            "site": {
                "type": "string",
                "enum": list(SITES.keys()),
                "description": "Qual site abrir",
            }
        },
        ["site"],
    ),
    _ferramenta(
        "pesquisar",
        "Faz uma pesquisa no Google sobre um termo ou assunto.",
        {"termo": {"type": "string", "description": "O que pesquisar"}},
        ["termo"],
    ),
    _ferramenta(
        "encerrar",
        "Coloca o Nino em modo de espera quando o usuário pede para desligar, parar ou encerrar a conversa.",
    ),
]


def executar_ferramenta(nome, argumentos, autorizada_visualmente=False):
    """Retorna (texto_para_falar, continuar_rodando)."""

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

    if nome == "abrir_aplicativo":
        return acao_abrir_aplicativo(argumentos.get("aplicativo", "")), True

    if nome == "abrir_pasta":
        return acao_abrir_pasta(argumentos.get("pasta", "")), True

    if nome == "criar_pasta":
        caminho = str(argumentos.get("caminho") or "").strip()
        caminho = os.path.expandvars(os.path.expanduser(caminho))
        if not caminho:
            return "Diga onde quer criar a pasta.", True
        try:
            os.makedirs(caminho, exist_ok=True)
        except OSError as erro:
            print(f"Falha ao criar pasta: {erro}")
            return "Não consegui criar essa pasta. Confira o caminho e as permissões.", True
        return f"Pasta pronta em {os.path.abspath(caminho)}.", True

    if nome == "criar_mesa_3d":
        caminho_token = os.path.join(
            os.environ.get("LOCALAPPDATA", os.path.expanduser("~")),
            "NinoX",
            "blender_token.txt",
        )
        try:
            with open(caminho_token, "r", encoding="utf-8") as arquivo:
                token = arquivo.read().strip()
            if not token:
                raise OSError("token vazio")
            dados = {
                "action": "create_table",
                "width": argumentos.get("largura", 1.2),
                "depth": argumentos.get("profundidade", 0.75),
                "diameter": argumentos.get("diametro"),
                "height": argumentos.get("altura", 0.75),
                "thickness": argumentos.get("espessura", 0.04),
                "name": argumentos.get("nome", "Mesa do Nino"),
            }
            requisicao = urllib.request.Request(
                "http://127.0.0.1:8765/command",
                data=json.dumps(dados).encode("utf-8"),
                headers={
                    "Authorization": f"Bearer {token}",
                    "Content-Type": "application/json",
                },
                method="POST",
            )
            with urllib.request.urlopen(requisicao, timeout=14) as resposta:
                resultado = json.loads(resposta.read().decode("utf-8"))
            return f"Mesa criada no Blender com {resultado.get('objects', 0)} peças.", True
        except urllib.error.HTTPError as erro:
            if erro.code == 401:
                return "A chave local da ponte Blender não corresponde. Reative o complemento NinoX no Blender.", True
            return f"O Blender recusou o pedido (HTTP {erro.code}). Confira se a cena está disponível.", True
        except (OSError, urllib.error.URLError, TimeoutError) as erro:
            print(f"Falha na ponte Blender: {erro}")
            return "Não encontrei a ponte do Blender. Instale e ative o complemento NinoX Local Bridge.", True
        except (ValueError, TypeError) as erro:
            print(f"Resposta inválida da ponte Blender: {erro}")
            return "A ponte do Blender retornou uma resposta inválida.", True

    if nome == "controlar_tela":
        texto = executar_controle_tela(
            argumentos.get("tarefa", ""),
            autorizada_visualmente,
            falar,
            ouvir,
        )
        return texto, True

    if nome == "abrir_site":
        return acao_abrir_site(argumentos.get("site", "")), True

    if nome == "pesquisar":
        return acao_pesquisar(argumentos.get("termo", "")), True

    if nome == "encerrar":
        global MODO_SUSPENSO
        MODO_SUSPENSO = True
        return "Tudo bem. Vou ficar em espera. Diga Nino para me chamar novamente.", True

    return "Não sei executar essa ação.", True


# ============================================================
# REGRAS LOCAIS (último recurso, sem IA e sem internet)
# ============================================================

_VERBOS_PESQUISA = ["pesquisar", "pesquise", "pesquisa", "procure", "busque", "buscar"]
_LIGACOES = ["por", "sobre", "no", "na", "google", "internet", "para", "pra", "de", "a", "o"]


def interpretar_localmente(comando):
    """Entende comandos simples só com palavras-chave.
    Retorna (nome_da_ferramenta, argumentos) ou None."""

    palavras = comando.lower().replace(",", " ").replace("?", " ").split()
    conjunto = set(palavras)

    frase_normalizada = normalizar_texto(comando)
    pedido_pasta = re.search(
        r"\b(?:crie|criar|cria)\s+(?:uma\s+)?pasta\s+(?:chamada\s+)?(.+)$",
        frase_normalizada,
    )
    if pedido_pasta:
        destino = pedido_pasta.group(1).strip()
        base = os.path.expanduser("~")
        for sufixo, pasta_usuario in (
            (" na area de trabalho", "Desktop"),
            (" no desktop", "Desktop"),
            (" em downloads", "Downloads"),
            (" nos downloads", "Downloads"),
            (" em documentos", "Documents"),
            (" nos documentos", "Documents"),
        ):
            if destino.endswith(sufixo):
                destino = destino[:-len(sufixo)].strip()
                base = os.path.join(base, pasta_usuario)
                break
        destino = re.sub(r"^(?:chamada|com nome)\s+", "", destino).strip()
        if destino:
            return "criar_pasta", {"caminho": os.path.join(base, destino)}

    if re.search(r"\bblender\b", frase_normalizada) and re.search(
        r"\b(mesa|mesinha)\b", frase_normalizada
    ) and re.search(
        r"\b(crie|criar|cria|modele|modelar|faz|faca|fazer)\b",
        frase_normalizada,
    ):
        medidas = {"largura": 1.2, "profundidade": 0.75, "altura": 0.75, "espessura": 0.04}
        redonda = bool(re.search(r"\b(redonda|diametro)\b", frase_normalizada))
        if redonda:
            medidas["diametro"] = 1.0
        texto_medidas = unicodedata.normalize("NFKD", comando.lower())
        texto_medidas = "".join(
            caractere for caractere in texto_medidas
            if not unicodedata.combining(caractere)
        )
        texto_medidas = re.sub(r"(?<=\d),(?=\d)", ".", texto_medidas)
        for chave, padrao in (
            ("largura", r"largura"),
            ("profundidade", r"profundidade"),
            ("diametro", r"diametro"),
            ("altura", r"altura"),
            ("espessura", r"espessura"),
        ):
            medida = re.search(
                rf"\b{padrao}\b\s+(?:de\s+|com\s+)?(\d+(?:[.,]\d+)?)\s*(cm|centimetros|m|metros)?",
                texto_medidas,
            )
            if medida:
                valor = float(medida.group(1).replace(",", "."))
                unidade = medida.group(2)
                em_centimetros = unidade in ("cm", "centimetros")
                inferir_centimetros = not unidade and valor > 10
                medidas[chave] = (
                    valor / 100
                    if em_centimetros or inferir_centimetros
                    else valor
                )
        medidas["nome"] = "Mesa do Nino"
        return "criar_mesa_3d", medidas

    if conjunto & {"sair", "encerrar", "desligar", "tchau"}:
        return "encerrar", {}

    for i, p in enumerate(palavras):
        if p in _VERBOS_PESQUISA:
            termo = palavras[i + 1:]
            while termo and termo[0] in _LIGACOES:
                termo = termo[1:]
            if termo:
                return "pesquisar", {"termo": " ".join(termo)}
            return None

    if "calculadora" in conjunto:
        return "abrir_calculadora", {}

    if "bloco" in conjunto or "notepad" in conjunto:
        return "abrir_bloco_notas", {}

    if conjunto & {"arquivos", "explorador"}:
        return "abrir_arquivos", {}

    if "paint" in conjunto or "desenho" in conjunto:
        return "abrir_aplicativo", {"aplicativo": "paint"}

    if "tarefas" in conjunto or "taskmgr" in conjunto:
        return "abrir_aplicativo", {"aplicativo": "gerenciador de tarefas"}

    if "configurações" in conjunto or "configuracoes" in conjunto:
        return "abrir_aplicativo", {"aplicativo": "configurações"}

    if conjunto & {"downloads", "documentos"}:
        pasta = "downloads" if "downloads" in conjunto else "documentos"
        return "abrir_pasta", {"pasta": pasta}

    if "trabalho" in conjunto or "desktop" in conjunto:
        return "abrir_pasta", {"pasta": "área de trabalho"}

    if conjunto & {"tela", "monitor", "janela"}:
        return "controlar_tela", {"tarefa": comando}

    for site in SITES:
        if site in conjunto:
            return "abrir_site", {"site": site}

    if conjunto & {"horas", "hora"}:
        return "dizer_hora", {}

    if "data" in conjunto or ("dia" in conjunto and "hoje" in conjunto):
        return "dizer_data", {}

    return None


def pedido_visual_direto(comando):
    """Só considera autorizada a tela quando o usuário pede para vê-la ou operá-la."""
    palavras = set(re.findall(r"\w+", comando.lower()))
    verbos_visuais = {
        "veja", "ver", "olha", "olhe", "olhar", "mostra", "mostrar",
        "leia", "ler", "analise", "analisar", "capture", "capturar",
    }
    verbos_controle = {
        "clique", "clicar", "digite", "digitar", "selecione", "selecionar",
    }
    referencias_tela = {"tela", "monitor", "janela"}
    pergunta_sobre_tela = bool(
        re.search(r"\b(o que|que)\b.{0,35}\b(tela|monitor|janela)\b", comando.lower())
    )
    return bool(
        palavras & verbos_controle
        or ((palavras & verbos_visuais) and (palavras & referencias_tela))
        or pergunta_sobre_tela
    )


def aplicativo_pedido_para_abrir(comando):
    frase = normalizar_texto(comando)
    if re.search(r"\b(nao|nunca)\b.{0,30}\b(abrir|abra|abre|abro|abri|iniciar|inicie|executar|execute|rodar|jogar)\b", frase):
        return None
    verbos = r"abrir|abra|abre|abro|abri|iniciar|inicie|inicia|executar|execute|executa|rodar|rode|roda|jogar|jogue"
    locais = r"na minha area de trabalho|na area de trabalho|no desktop|na desktop|no menu iniciar|pelo menu iniciar|no computador"
    aplicativos_conhecidos = {
        *(normalizar_texto(nome) for nome in APLICATIVOS),
        *(normalizar_texto(nome) for nome in SITES),
        "calculadora", "bloco de notas", "notepad", "arquivos",
        "explorador de arquivos", "downloads", "documentos", "area de trabalho",
    }
    busca_na_tela = bool(re.search(rf"\b(?:{locais})\b|\batalho\b", frase))
    complementos_local = re.compile(
        r"\b(?:no meu computador|no computador|do meu computador|no meu pc|"
        r"no pc|no windows|pelo computador|no menu iniciar|pela busca do menu iniciar)\b"
    )

    for ocorrencia in re.finditer(rf"\b(?:{verbos})\b", frase):
        alvo = frase[ocorrencia.end():]
        alvo = re.split(
            rf"\b(?:{locais}|e depois|depois|e me diga|e me avise|quando terminar)\b",
            alvo,
            maxsplit=1,
        )[0]
        # Remove descrições que o reconhecimento de voz pode anexar ao nome
        # do aplicativo, por exemplo: "Valorant que está instalado no meu PC".
        alvo = re.split(
            r"\b(?:que (?:esta|foi|fica|funciona)|instalad[oa]s?)\b",
            alvo,
            maxsplit=1,
        )[0]
        alvo = complementos_local.split(alvo, maxsplit=1)[0]
        proxima_acao = re.search(rf"\b(?:{verbos})\b", alvo)
        if proxima_acao:
            alvo = alvo[:proxima_acao.start()]
        alvo = alvo.strip()
        alvo = re.sub(
            r"^(?:(?:o|a|os|as|um|uma|meu|minha|aplicativo|programa|jogo)\s+)+",
            "",
            alvo,
        )
        alvo = re.sub(r"\b(por favor|pra mim|para mim)\b", "", alvo).strip()
        alvo = re.sub(r"\s+(?:por favor|pra mim|para mim)$", "", alvo).strip()

        if not alvo or (alvo in aplicativos_conhecidos and not busca_na_tela):
            continue
        if alvo in {"valorante", "valorant"}:
            return "VALORANT"
        return alvo

    return None


def aplicativo_pedido_na_area_de_trabalho(comando):
    """Extrai o app quando o usuário pede para mostrar a Área de Trabalho e abri-lo."""
    frase = normalizar_texto(comando)
    if not re.search(r"\b(area de trabalho|desktop)\b", frase):
        return None

    verbos = r"abrir|abra|abre|abro|abri|iniciar|inicie|inicia|executar|execute|executa|va|vai|ir"
    ocorrencias = list(re.finditer(rf"\b(?:{verbos})\b", frase))
    if len(ocorrencias) < 2:
        return None

    alvo = frase[ocorrencias[-1].end():]
    alvo = re.split(
        r"\b(?:no meu computador|no computador|do meu computador|no meu pc|no pc)\b",
        alvo,
        maxsplit=1,
    )[0].strip()
    alvo = re.sub(r"^(?:(?:e|depois|o|a|os|as|um|uma|meu|minha|aplicativo|programa|jogo)\s+)+", "", alvo)
    alvo = re.sub(r"\b(por favor|pra mim|para mim)\b", "", alvo).strip()
    if alvo in {"valorante", "valorant"}:
        return "VALORANT"
    return alvo or None


def pedido_interacao_visual(comando):
    frase = normalizar_texto(comando)
    if re.search(r"\b(nao|nunca)\b.{0,30}\b(clique|clica|clicar|digite|digitar|pesquise|pesquisar|abra|abrir)\b", frase):
        return False

    operacoes = re.findall(
        r"\b(abrir|abra|abre|iniciar|inicie|inicia|va|vai|entre|entra|localize|localizar|encontre|encontrar|"
        r"clique|clica|clicar|digite|digita|digitar|escreva|escreve|escrever|"
        r"pesquise|pesquisa|pesquisar|procure|procurar|busque|buscar|selecione|selecionar|"
        r"envie|envia|enviar|mande|manda|mandar|"
        r"toque|pressione|pressionar|role|rolar|feche|fechar|minimize|minimizar|"
        r"maximize|maximizar|salve|salvar|copie|copiar|cole|colar)\b",
        frase,
    )
    if len(operacoes) >= 2:
        return True

    menciona_interface = bool(
        re.search(
            r"\b(youtube|whatsapp|instagram|facebook|chrome|firefox|edge|discord|spotify|"
            r"valorant|valorante|steam|contato|aplicativo|app|programa|site|navegador|browser|"
            r"mensagem|computador|pc|tela|janela|desktop|area de trabalho|documento|arquivo|pasta|menu|botao|campo)\b",
            frase,
        )
    )
    menciona_acao = bool(
        re.search(
            r"\b(clique|clica|clicar|digite|digita|digitar|pesquise|pesquisa|pesquisar|"
            r"selecione|envie|enviar|mande|mandar|role|rolar|feche|fechar|minimize|minimizar|"
            r"maximize|maximizar|salve|salvar|copie|copiar|cole|colar|abra|abrir|abre)\b",
            frase,
        )
    )
    contato_no_whatsapp = bool(
        "whatsapp" in frase
        and re.search(r"\b(contato|conversa|chat)\b", frase)
        and re.search(r"\b(abrir|abra|abre|entrar|entre|entra|localize|encontre)\b", frase)
    )
    pedido_mensagem = bool(
        re.search(r"\b(manda|mande|mandar|envia|envie|enviar)\b.{0,60}\bmensagem\b", frase)
    )
    pedido_video_por_posicao = bool(
        re.search(
            r"\b(?:primeiro|primeira|segundo|segunda)\b.{0,24}\b(?:video|resultado)\b|"
            r"\b(?:video|resultado)\b.{0,24}\b(?:primeiro|primeira|segundo|segunda)\b",
            frase,
        )
        and re.search(r"\b(clique|clica|clicar|abra|abre|abrir|selecione)\b", frase)
    )
    return (
        (menciona_interface and menciona_acao)
        or contato_no_whatsapp
        or pedido_mensagem
        or pedido_video_por_posicao
    )


# ============================================================
# PERSONALIDADE DO NINO
# ============================================================

INSTRUCOES_SISTEMA = (
    "Você é Nino, um assistente virtual pessoal que responde em português do Brasil. "
    "Sua personalidade é calorosa, expressiva, inteligente e prestativa, com humor leve quando combinar. "
    "Se o usuário fizer uma piada, reaja com naturalidade e às vezes com um 'haha' breve; não force risadas, "
    "não faça piadas em momentos sérios e não afirme ter sentimentos humanos. "
    "Use as preferências salvas para adaptar seu jeito de conversar sem perder clareza. "
    "Quando o pedido do usuário corresponder a uma ação conhecida, use a ferramenta apropriada. "
    "Use controlar_tela quando a tarefa exigir observar a tela. A autorização contínua de controle já foi concedida pelo usuário. "
    "A ferramenta de tela disponível se chama exatamente controlar_tela. Nunca invente outro nome de ferramenta. "
    "Não volte a pedir autorização para tarefas comuns de tela; pergunte quando a ordem ou o alvo estiver ambíguo. "
    "As ações disponíveis incluem controlar a tela, abrir qualquer aplicativo instalado, abrir ou criar pastas, "
    "criar uma mesa 3D no Blender, abrir sites, pesquisar algo, "
    "Use criar_mesa_3d quando o usuário pedir para modelar uma mesa no Blender; a ponte local cria o tampo e quatro pés com medidas em metros. "
    "informar hora, informar data e colocar a conversa em espera até o usuário chamar Nino novamente. "
    "Se for uma pergunta geral, um pedido de conversa ou algo sem ferramenta correspondente, "
    "responda em texto de forma breve, educada e objetiva, em no máximo duas frases, "
    "pois a resposta será falada em voz alta. "
    "O texto do usuário vem de reconhecimento de voz e pode conter pequenos erros."
)


# ============================================================
# CÉREBRO: provedores locais e online, selecionáveis por voz/configuração
# ============================================================

def criar_clientes():
    """Descobre provedores disponíveis sem iniciar downloads ou serviços."""

    clientes = []

    servidores_locais = (
        (
            "OpenJarvis",
            OPENJARVIS_URL,
            os.environ.get("OPENJARVIS_API_KEY", "local"),
            os.environ.get("OPENJARVIS_MODEL"),
        ),
        (
            "Ollama",
            OLLAMA_URL,
            "local",
            os.environ.get("NINO_LOCAL_CHAT_MODEL"),
        ),
    )
    for nome, url, chave, modelo_preferido in servidores_locais:
        try:
            cliente = OpenAI(
                api_key=chave,
                base_url=url,
                timeout=60 if nome == "Ollama" else 30,
                max_retries=0,
            )
            modelos = cliente.with_options(timeout=2).models.list().data
            ids_modelos = [modelo.id for modelo in modelos]
            if nome == "Ollama":
                modelo_padrao = next(
                    (identificador for identificador in ids_modelos
                     if identificador == "qwen3.5:4b"),
                    None,
                )
                modelo_escolhido = (
                    modelo_preferido
                    if modelo_preferido in ids_modelos
                    else modelo_padrao
                    or (
                        os.environ.get("NINO_VISION_MODEL")
                        if os.environ.get("NINO_VISION_MODEL") in ids_modelos
                        else None
                    )
                    or next(
                        (identificador for identificador in ids_modelos
                         if "embed" not in identificador.lower()),
                        None,
                    )
                )
            else:
                modelo_escolhido = modelo_preferido or (
                    ids_modelos[0] if ids_modelos else None
                )
            if modelo_escolhido:
                clientes.append((nome, "local", cliente, modelo_escolhido))
                print(f"✅ {nome} local disponível (modelo: {modelo_escolhido})")
            else:
                print(f"ℹ️ {nome} respondeu sem modelos disponíveis.")
        except Exception:
            print(f"ℹ️ {nome} não encontrado em {url}")

    # Provedores online só são usados nos modos online/automático.
    chave_gemini = os.environ.get("GEMINI_API_KEY")
    if chave_gemini:
        gemini = OpenAI(
            api_key=chave_gemini,
            base_url=GEMINI_URL,
            timeout=30,
            max_retries=0,
        )
        clientes.append(("Gemini", "online", gemini, MODELO_GEMINI))
        print(f"✅ Gemini disponível quando o modo online for selecionado ({MODELO_GEMINI})")
    else:
        print("ℹ️ GEMINI_API_KEY não encontrada.")

    chave_groq = os.environ.get("GROQ_API_KEY")
    if chave_groq:
        groq = OpenAI(
            api_key=chave_groq,
            base_url=GROQ_URL,
            timeout=30,
            max_retries=0,
        )
        clientes.append(("Groq", "online", groq, MODELO_GROQ))
        print("✅ Groq disponível quando o modo online for selecionado.")
    else:
        print("ℹ️ GROQ_API_KEY não encontrada.")

    return clientes


def limpar_resposta(texto):
    """Remove blocos <think>...</think> que alguns modelos locais enviam."""
    if not texto:
        return ""
    texto = re.sub(r"<think>.*?</think>", "", texto, flags=re.DOTALL)
    return texto.strip()


def comando_precisa_de_ferramentas(comando):
    """Inclui ferramentas apenas quando a fala provavelmente pede uma ação."""
    frase = normalizar_texto(comando)
    padrao_acao = (
        r"\b(abrir|abra|abre|abro|iniciar|inicie|inicia|criar|crie|salvar|salve|"
        r"pesquisar|pesquise|procurar|procure|clicar|clique|digitar|digite|"
        r"escrever|escreva|apagar|apague|mover|mova|controlar|controle|"
        r"ver|veja|olhar|olhe|analisar|analise|capturar|captura|"
        r"executar|execute|rodar|rode|fazer|faca|consultar|consulte)\b"
        r"|\b(tela|monitor|janela|aplicativo|programa|pasta|arquivo|site|youtube|"
        r"whatsapp|blender|valorant|hora|data|clima|tempo)\b"
    )
    return bool(re.search(padrao_acao, frase))


def consultar_ia(comando):
    """Escolhe provedores conforme a preferência e retorna a primeira resposta."""
    global PROVEDOR_ATUAL, ULTIMO_ERRO_IA

    ULTIMO_ERRO_IA = None

    modo = carregar_configuracao()["modo_ia"]
    locais = [cliente for cliente in CLIENTES if cliente[1] == "local"]
    online = [cliente for cliente in CLIENTES if cliente[1] == "online"]
    if modo == "local":
        ordem = locais
    elif modo == "online":
        ordem = online + locais
    else:
        ordem = locais + online

    mensagens = (
        [{"role": "system", "content": INSTRUCOES_SISTEMA + contexto_memoria()}]
        + historico
        + [{"role": "user", "content": comando}]
    )

    falha_online = False
    for nome, grupo, cliente, nome_modelo in ordem:
        aviso_previo = False
        try:
            if grupo == "online" and nome != PROVEDOR_ATUAL:
                mostrar_aviso(
                    f"Nino vai usar {nome} online. Este comando será enviado pela internet."
                )
                time.sleep(0.4)
                aviso_previo = True
            opcoes = {}
            if comando_precisa_de_ferramentas(comando):
                opcoes["tools"] = FERRAMENTAS

            resposta = cliente.chat.completions.create(
                model=nome_modelo,
                messages=mensagens,
                **opcoes,
            )
            print(f"(cérebro: {nome})")
            if nome != PROVEDOR_ATUAL:
                fallback = grupo == "local" and (
                    modo == "online" or (modo == "automatico" and falha_online)
                )
                if not aviso_previo and not fallback and (
                    PROVEDOR_ATUAL is not None or grupo == "online"
                ):
                    detalhe = "modo online" if grupo == "online" else "modo local"
                    mostrar_aviso(f"Nino usando {nome} · {detalhe}")
                PROVEDOR_ATUAL = nome
            if grupo == "online":
                falha_online = False
            elif (
                modo == "online"
                or (modo == "automatico" and falha_online)
            ) and grupo == "local":
                definir_modo("local")
                mostrar_aviso("O provedor online falhou. Nino voltou ao modo local.")
            return resposta.choices[0].message

        except Exception as erro:
            print(f"Falha no cérebro {nome}: {erro}")
            texto_erro = str(erro).lower()
            if (
                isinstance(erro, TimeoutError)
                or "timeout" in texto_erro
                or "timed out" in texto_erro
            ):
                ULTIMO_ERRO_IA = "timeout"
            falha_online = falha_online or grupo == "online"

    if falha_online and modo == "automatico":
        definir_modo("local")
        mostrar_aviso("Os provedores online falharam. Nino voltou ao modo local.")

    return None


def modo_pedido_por_voz(comando):
    """Converte pedidos falados simples em uma preferência de provedor."""
    frase = normalizar_texto(comando)
    if not re.search(r"\b(modo|mude|mudar|troque|trocar|use|usar|quero)\b", frase):
        return None
    if re.search(r"\b(local|offline|gratis|gratuito)\b", frase):
        return "local"
    if re.search(r"\b(automatico|automatica)\b", frase):
        return "automatico"
    if re.search(r"\b(online|nuvem|gemini|groq)\b", frase):
        return "online"
    return None


def lembrar(usuario, nino):
    historico.append({"role": "user", "content": usuario})
    historico.append({"role": "assistant", "content": nino})

    while len(historico) > HISTORICO_MAX:
        historico.pop(0)


def executar_tarefa_visual(tarefa, autorizada_visualmente=False):
    """Executa a tela e guarda apenas falhas que ocorreram antes de qualquer ação."""
    global TAREFA_TELA_REPETIR

    texto, continuar = executar_ferramenta(
        "controlar_tela",
        {"tarefa": tarefa},
        autorizada_visualmente=autorizada_visualmente,
    )
    if texto.startswith("Não consegui analisar a tela antes da primeira ação:"):
        TAREFA_TELA_REPETIR = (tarefa, autorizada_visualmente)
    else:
        TAREFA_TELA_REPETIR = None
    return texto, continuar


def processar_comando(comando):
    global TAREFA_TELA_PENDENTE, TAREFA_TELA_REPETIR, MODO_SUSPENSO, ULTIMO_ERRO_IA

    if not comando:
        return True

    frase_normalizada = normalizar_texto(comando)
    comandos_repetir = {
        "tente novamente", "tenta novamente", "tente de novo", "tenta de novo",
        "repita", "repete", "repita a tarefa", "repete a tarefa",
    }
    if frase_normalizada in comandos_repetir and TAREFA_TELA_REPETIR:
        tarefa, autorizada = TAREFA_TELA_REPETIR
        texto, continuar = executar_tarefa_visual(tarefa, autorizada)
        falar(texto)
        lembrar(comando, texto)
        return continuar
    TAREFA_TELA_REPETIR = None

    modo_solicitado = modo_pedido_por_voz(comando)
    if modo_solicitado:
        modo_anterior = carregar_configuracao()["modo_ia"]
        if modo_anterior != modo_solicitado:
            definir_modo(modo_solicitado)
            rotulos = {
                "local": "local e gratuito",
                "online": "online",
                "automatico": "automático",
            }
            mostrar_aviso(f"Modo do Nino: {rotulos[modo_solicitado]}")
            print(f"Modo de IA alterado para: {rotulos[modo_solicitado]}.")
        else:
            mostrar_aviso("Esse modo já está selecionado.")
        return True

    try:
        resposta_memoria = processar_comando_memoria(comando)
    except OSError as erro:
        print(f"Falha ao salvar memória local: {erro}")
        resposta_memoria = "Não consegui salvar isso na memória local."
    if resposta_memoria:
        falar(resposta_memoria)
        lembrar(comando, resposta_memoria)
        return True

    if comando_para_suspender(comando):
        MODO_SUSPENSO = True
        resposta_final = "Tudo bem. Vou ficar em espera. Diga Nino para me chamar novamente."
        falar(resposta_final)
        lembrar(comando, resposta_final)
        return True

    # Uma resposta curta de autorização pertence ao pedido visual anterior;
    # resolve localmente para não depender de o Groq manter esse contexto.
    if TAREFA_TELA_PENDENTE:
        confirmacao = interpretar_confirmacao(comando)
        if confirmacao is False:
            TAREFA_TELA_PENDENTE = None
            resposta_final = "Tudo bem. Não acessei sua tela."
            falar(resposta_final)
            lembrar(comando, resposta_final)
            return True
        if confirmacao is True:
            tarefa = TAREFA_TELA_PENDENTE
            TAREFA_TELA_PENDENTE = None
            texto, continuar = executar_tarefa_visual(tarefa, True)
            falar(texto)
            lembrar(comando, texto)
            return continuar

    pedido_fechamento = pedido_fechar_janela(comando)
    if pedido_fechamento:
        texto = fechar_janela_local(*pedido_fechamento)
        falar(texto)
        lembrar(comando, texto)
        return True

    termo_youtube = pesquisa_youtube_pedida(comando)
    if termo_youtube:
        texto = pesquisar_youtube(termo_youtube)
        falar(texto)
        lembrar(comando, texto)
        return True

    if pedido_abrir_whatsapp(comando):
        acao_abrir_site("whatsapp")
        texto = "Certo, abrindo o WhatsApp Web."
        falar(texto)
        lembrar(comando, texto)
        return True

    aplicativo_area_trabalho = aplicativo_pedido_na_area_de_trabalho(comando)
    if aplicativo_area_trabalho:
        texto = abrir_aplicativo_menu_iniciar(
            aplicativo_area_trabalho,
            mostrar_area_trabalho=True,
        )
        falar(texto)
        lembrar(comando, texto)
        return True

    aplicativo = aplicativo_pedido_para_abrir(comando)
    quantidade_acoes = len(re.findall(
        r"\b(abrir|abra|abre|abro|abri|iniciar|inicie|inicia|vai|va|clique|clicar|digite|digitar|"
        r"pesquise|pesquisar|procure|procurar|selecione|selecionar)\b",
        normalizar_texto(comando),
    ))
    if aplicativo and quantidade_acoes <= 1:
        texto = abrir_aplicativo_menu_iniciar(aplicativo)
        falar(texto)
        lembrar(comando, texto)
        return True

    if pedido_interacao_visual(comando):
        tarefa = (
            "Execute a sequência completa pedida pelo usuário usando a interface visual. "
            "Se ele pediu para abrir e depois pesquisar, clicar ou digitar, faça também as etapas seguintes. "
            "Confira a tela após cada ação e continue até concluir o pedido. "
            "No YouTube, pesquise o vídeo solicitado. Se o usuário pedir um resultado por posição, "
            "conte os vídeos principais de cima para baixo, ignore anúncios e a barra lateral, "
            "e clique exatamente no primeiro ou segundo vídeo solicitado. Confira a posição antes do clique. "
            "Para pedidos de mensagem sem aplicativo especificado, use o WhatsApp que estiver aberto. "
            "No WhatsApp, localize o contato pelo nome exato e digite a mensagem solicitada. "
            "Digitar uma mensagem não significa enviá-la: só envie se o usuário pedir explicitamente. "
            "Se a ordem explícita incluir o envio e o contato estiver claro, execute-a; pergunte apenas se houver ambiguidade. "
            f"Pedido original: {comando}"
        )
        texto, continuar = executar_tarefa_visual(
            tarefa,
            autorizada_visualmente=pedido_visual_direto(comando),
        )
        falar(texto)
        lembrar(comando, texto)
        return continuar

    if aplicativo:
        texto = abrir_aplicativo_menu_iniciar(aplicativo)
        falar(texto)
        lembrar(comando, texto)
        return True

    mensagem = consultar_ia(comando)
    autorizada_visualmente = pedido_visual_direto(comando)

    continuar = True
    falado = []

    # --- Caso 1: a IA respondeu ---
    if mensagem is not None:

        if mensagem.tool_calls:
            for chamada in mensagem.tool_calls:
                try:
                    argumentos = (
                        json.loads(chamada.function.arguments)
                        if chamada.function.arguments
                        else {}
                    )
                except json.JSONDecodeError:
                    argumentos = {}

                texto, continuar = executar_ferramenta(
                    chamada.function.name,
                    argumentos,
                    autorizada_visualmente=autorizada_visualmente,
                )
                falado.append(texto)

        else:
            texto = limpar_resposta(mensagem.content)
            if re.search(
                r"\b(posso|quer que eu|preciso)\b.{0,80}\b(acessar|ver|olhar|analisar|controlar)\b.{0,50}\b(tela|monitor|janela)\b",
                texto.lower(),
            ):
                # O modelo às vezes pergunta em texto em vez de chamar a ferramenta.
                # Guarda a tarefa para o próximo "sim" executar no fluxo correto.
                TAREFA_TELA_PENDENTE = comando
            falado.append(texto or "Não consegui formular uma resposta.")

    # --- Caso 2: nenhuma IA disponível -> regras locais ---
    else:
        acao = interpretar_localmente(comando)

        if acao:
            texto, continuar = executar_ferramenta(
                acao[0],
                acao[1],
                autorizada_visualmente=autorizada_visualmente,
            )
            falado.append(texto)
        else:
            if ULTIMO_ERRO_IA == "timeout":
                falado.append(
                    "O modelo local demorou para responder. Pode repetir em uma frase mais curta?"
                )
            else:
                falado.append(
                    "Estou sem acesso à minha inteligência e não entendi esse comando."
                )

    resposta_final = " ".join(falado)
    falar(resposta_final)
    lembrar(comando, resposta_final)

    return continuar


# ============================================================
# PROGRAMA PRINCIPAL
# ============================================================

def main():
    global modelo, modelo_whisper, CLIENTES, MODO_SUSPENSO

    print("Carregando modelo de português...")

    try:
        modelo = Model(MODELO)
    except Exception as erro:
        print("Erro ao carregar o modelo Vosk.")
        print(erro)
        sys.exit(1)

    print(f"Carregando Faster-Whisper '{MODELO_WHISPER}' em CPU (INT8)...")
    try:
        modelo_whisper = WhisperModel(
            MODELO_WHISPER,
            device="cpu",
            compute_type="int8",
            cpu_threads=min(8, os.cpu_count() or 4),
        )
        print("✅ Faster-Whisper small pronto para a conversa.")
    except Exception as erro:
        modelo_whisper = None
        print(f"⚠️ Faster-Whisper não iniciou; usando Vosk como alternativa: {erro}")

    CLIENTES = criar_clientes()

    if not CLIENTES:
        print("⚠️ Nenhuma IA disponível: o Nino vai usar só comandos simples.")

    falar("Sistemas iniciados. Pode falar comigo; estou ouvindo.")

    while True:
        if MODO_SUSPENSO:
            print("\n💤 Nino em espera. Diga 'Nino' para retomar.")
            comando = ouvir(espera_ativacao=True)
            if not comando or not detectar_nino(comando):
                continue

            MODO_SUSPENSO = False
            complemento = extrair_comando(comando)
            if pergunta_se_nino_esta_ativo(complemento):
                falar("Estou aqui. Pode falar comigo normalmente.")
                continue
            falar("Estou aqui. Pode falar comigo.")
            continue

        print("\n🎙️ Conversa por voz ativa. Estou ouvindo...")
        try:
            comando = fila_comandos.get_nowait()
        except queue.Empty:
            comando = ouvir()
        if not comando:
            continue

        if not processar_comando(comando):
            break


if __name__ == "__main__":
    main()
