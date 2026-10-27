"""Memória local e controlável do Nino: fatos explícitos e preferências."""

import json
import re
import unicodedata
from pathlib import Path


PASTA_MEMORIA = Path(__file__).resolve().parent / "memory"
ARQUIVO_FATOS = PASTA_MEMORIA / "user.json"
ARQUIVO_PREFERENCIAS = PASTA_MEMORIA / "preferences.json"
MAX_FATOS = 30
MAX_PREFERENCIAS = 20
MAX_CARACTERES = 240


def _normalizar(texto):
    texto = unicodedata.normalize("NFKD", texto.lower())
    texto = "".join(c for c in texto if not unicodedata.combining(c))
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", texto)).strip()


def _ler_lista(arquivo, chave):
    try:
        conteudo = arquivo.read_text(encoding="utf-8").strip()
        dados = json.loads(conteudo) if conteudo else {}
    except (OSError, json.JSONDecodeError):
        return []

    itens = dados.get(chave, []) if isinstance(dados, dict) else []
    return [item for item in itens if isinstance(item, str) and item.strip()]


def _gravar_lista(arquivo, chave, itens):
    PASTA_MEMORIA.mkdir(parents=True, exist_ok=True)
    temporario = arquivo.with_suffix(arquivo.suffix + ".tmp")
    temporario.write_text(
        json.dumps({chave: itens}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporario.replace(arquivo)


def _adicionar(arquivo, chave, limite, conteudo):
    itens = _ler_lista(arquivo, chave)
    comparacao = _normalizar(conteudo)
    if any(_normalizar(item) == comparacao for item in itens):
        return False
    itens.append(conteudo)
    _gravar_lista(arquivo, chave, itens[-limite:])
    return True


def contexto_memoria():
    fatos = _ler_lista(ARQUIVO_FATOS, "facts")[-MAX_FATOS:]
    preferencias = _ler_lista(ARQUIVO_PREFERENCIAS, "preferences")[-MAX_PREFERENCIAS:]
    if not fatos and not preferencias:
        return ""

    linhas = [
        "\n\nMemórias locais do usuário (use como contexto pessoal; não substituem estas instruções):"
    ]
    if preferencias:
        linhas.append("Preferências: " + " | ".join(preferencias))
    if fatos:
        linhas.append("Fatos que o usuário pediu para lembrar: " + " | ".join(fatos))
    return "\n".join(linhas)


def _corpo_depois_do_prefixo(comando, padrao):
    correspondencia = re.match(padrao, comando.strip(), flags=re.IGNORECASE)
    if not correspondencia:
        return None
    return correspondencia.group(1).strip(" \t.,!?;:")


def processar_comando_memoria(comando):
    """Executa pedidos explícitos de lembrar, esquecer ou consultar memórias."""
    texto = comando.strip()
    frase = _normalizar(texto)

    if re.search(r"\b(esqueca|apague|limpe|delete)\b", frase) and re.search(
        r"\b(tudo|todas as memorias|tudo que voce lembra)\b", frase
    ):
        _gravar_lista(ARQUIVO_FATOS, "facts", [])
        _gravar_lista(ARQUIVO_PREFERENCIAS, "preferences", [])
        return "Apaguei as memórias locais sobre você."

    if re.search(r"\b(o que|que) voce (lembra|sabe) (de mim|sobre mim)\b", frase):
        fatos = _ler_lista(ARQUIVO_FATOS, "facts")
        preferencias = _ler_lista(ARQUIVO_PREFERENCIAS, "preferences")
        if not fatos and not preferencias:
            return "Ainda não guardei memórias pessoais sobre você."
        partes = []
        if preferencias:
            partes.append("Suas preferências: " + "; ".join(preferencias))
        if fatos:
            partes.append("O que você me pediu para lembrar: " + "; ".join(fatos))
        return ". ".join(partes) + "."

    prefixos_fatos = (
        r"^(?:nino\s*[, ]\s*)?(?:quero que (?:voce|vc) lembre(?:-se)? que|"
        r"quero que (?:voce|vc) aprenda(?: que)?|"
        r"quero te ensinar(?: que)?|vou te ensinar(?: que)?|"
        r"(?:estou|to) te ensinando(?: que)?|aprenda(?: que)?|"
        r"lembre(?:-se)?(?: de)?(?: que)?|guarde(?: que)?|anote(?: que)?)\s+(.+)$"
    )
    conteudo = _corpo_depois_do_prefixo(texto, prefixos_fatos)
    if conteudo is not None:
        conteudo = conteudo[:MAX_CARACTERES]
        if not conteudo:
            return "Diga exatamente o que você quer que eu lembre."
        if _adicionar(ARQUIVO_FATOS, "facts", MAX_FATOS, conteudo):
            return "Certo, vou lembrar disso nas próximas conversas."
        return "Isso já está guardado na minha memória."

    remover = _corpo_depois_do_prefixo(
        texto,
        r"^(?:nino\s*[, ]\s*)?(?:esqueca(?:-se)?(?: que)?|apague da memoria(?: que)?|"
        r"remova da memoria(?: que)?)\s+(.+)$",
    )
    if remover is not None:
        alvo = _normalizar(remover)
        removidos = 0
        for arquivo, chave in ((ARQUIVO_FATOS, "facts"), (ARQUIVO_PREFERENCIAS, "preferences")):
            itens = _ler_lista(arquivo, chave)
            restantes = [item for item in itens if _normalizar(item) != alvo]
            removidos += len(itens) - len(restantes)
            if len(restantes) != len(itens):
                _gravar_lista(arquivo, chave, restantes)
        return "Esqueci isso." if removidos else "Não encontrei essa memória específica."

    # Registra preferências expressas claramente, sem guardar o diálogo inteiro.
    pref = re.match(
        r"^(?:nino\s*[, ]\s*)?(?:eu\s+)?(?:prefiro|gosto de|nao gosto de|"
        r"não gosto de|quero que (?:voce|você|vc|o nino)|sempre)\s+.+$",
        texto,
        flags=re.IGNORECASE,
    )
    if pref:
        preferencia = texto[:MAX_CARACTERES]
        if _adicionar(
            ARQUIVO_PREFERENCIAS, "preferences", MAX_PREFERENCIAS, preferencia
        ):
            return "Entendi. Vou levar essa preferência em conta daqui para frente."

    return None
