import json
import queue
import sys

import sounddevice as sd
from vosk import Model, KaldiRecognizer

MODELO = "vosk-model-small-pt-0.3"
MICROFONE = 2  # ajuste conforme seu dispositivo
TAXA = 16000

# Lista pequena e restrita: só isso o Vosk vai tentar reconhecer.
# Incluímos variações que ele costuma confundir com "jarvis".
GRAMATICA = json.dumps([
    "jarvis",
    "jarves",
    "jardim",
    "já vi",
    "[unk]"   # necessário: permite que ele "descarte" sons que não batem
])

print("Carregando modelo...")
modelo = Model(MODELO)
reconhecedor = KaldiRecognizer(modelo, TAXA, GRAMATICA)

fila_audio = queue.Queue()


def callback(indata, frames, tempo, status):
    if status:
        print(status, file=sys.stderr)
    fila_audio.put(bytes(indata))


print("Diga 'NINO' (Ctrl+C para sair)...\n")

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
                print(f"Reconhecido: '{texto}'")