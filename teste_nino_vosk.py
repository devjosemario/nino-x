import json
import queue
import sys
from collections import Counter

import sounddevice as sd
from vosk import Model, KaldiRecognizer

MODELO = "vosk-model-small-pt-0.3"
MICROFONE = 1   # use o mesmo índice que funciona no seu projeto
TAXA = 16000
GRAMATICA = json.dumps(
    ["nino", "oi nino", "ei nino", "ola nino", "[unk]"],
    ensure_ascii=False,
)

print("Carregando modelo...")
modelo = Model(MODELO)
reconhecedor = KaldiRecognizer(modelo, TAXA, GRAMATICA)

fila_audio = queue.Queue()
ouvidos = []


def callback(indata, frames, tempo, status):
    if status:
        print(status, file=sys.stderr)
    fila_audio.put(bytes(indata))


print("\nTeste com vocabulário restrito: diga 'Nino' e 'Oi, Nino'.")
print("Repita umas 10 vezes cada. Ctrl+C para encerrar e ver o resumo.\n")

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
                texto = json.loads(reconhecedor.Result()).get("text", "")
                if texto:
                    ouvidos.append(texto)
                    print(f"Ouvi: '{texto}'")
except KeyboardInterrupt:
    pass

print("\n=== RESUMO (copie e cole no chat) ===")
for texto, qtd in Counter(ouvidos).most_common():
    print(f"{qtd}x  {texto}")
