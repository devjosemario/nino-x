"""Preferências persistentes do Nino e avisos visuais leves."""

import json
import queue
import threading
from pathlib import Path


ARQUIVO_CONFIG = Path(__file__).resolve().parent / "config" / "nino_settings.json"
MODOS = {"local", "online", "automatico"}
_fila_avisos = queue.Queue()
_thread_avisos = None
_lock_avisos = threading.Lock()


def carregar_configuracao():
    """Lê as preferências; a instalação nova começa no modo local."""
    try:
        dados = json.loads(ARQUIVO_CONFIG.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        dados = {}

    modo = dados.get("modo_ia", "local") if isinstance(dados, dict) else "local"
    if modo not in MODOS:
        modo = "local"
    return {"modo_ia": modo}


def definir_modo(modo):
    """Salva o modo selecionado com substituição atômica do arquivo."""
    modo = modo.strip().lower()
    if modo not in MODOS:
        raise ValueError("Modo inválido. Use local, online ou automatico.")

    dados = carregar_configuracao()
    dados["modo_ia"] = modo
    ARQUIVO_CONFIG.parent.mkdir(parents=True, exist_ok=True)
    temporario = ARQUIVO_CONFIG.with_suffix(".json.tmp")
    temporario.write_text(
        json.dumps(dados, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporario.replace(ARQUIVO_CONFIG)
    return modo


def _loop_avisos():
    """Cria a janela Tk apenas quando houver um aviso a mostrar."""
    try:
        import tkinter as tk
    except ImportError:
        while True:
            texto = _fila_avisos.get()
            print(f"[Nino] {texto}")

    while True:
        texto = _fila_avisos.get()
        try:
            janela = tk.Tk()
            janela.overrideredirect(True)
            janela.attributes("-topmost", True)
            janela.configure(bg="#20242b")
            rotulo = tk.Label(
                janela,
                text=texto,
                bg="#20242b",
                fg="#f2f4f8",
                font=("Segoe UI", 10),
                padx=16,
                pady=11,
                wraplength=360,
                justify="left",
            )
            rotulo.pack()
            janela.update_idletasks()
            x = janela.winfo_screenwidth() - janela.winfo_width() - 24
            janela.geometry(f"+{x}+24")
            janela.after(4000, janela.destroy)
            janela.mainloop()
        except Exception:
            print(f"[Nino] {texto}")


def mostrar_aviso(texto):
    """Envia um aviso curto para uma janela de canto, sem bloquear o áudio."""
    global _thread_avisos
    if not texto:
        return
    with _lock_avisos:
        if _thread_avisos is None or not _thread_avisos.is_alive():
            _thread_avisos = threading.Thread(target=_loop_avisos, daemon=True)
            _thread_avisos.start()
    _fila_avisos.put(texto)
