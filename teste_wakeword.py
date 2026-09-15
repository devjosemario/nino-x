import numpy as np
import sounddevice as sd
from openwakeword.model import Model
import openwakeword

MICROFONE = 1          # ajuste conforme seu dispositivo
TAXA = 16000
TAMANHO_BLOCO = 1280    # openWakeWord espera blocos de 80ms (1280 amostras a 16kHz)

LIMIAR = 0.5            # confiança mínima para considerar ativação (0.0 a 1.0)


print("Baixando/carregando modelos pré-treinados...")
openwakeword.utils.download_models()

modelo = Model(wakeword_models=["hey_nino"])

print("Diga 'Hey NINO' (Ctrl+C para sair)...\n")


def callback(indata, frames, tempo, status):
    if status:
        print(status)

    audio = np.frombuffer(indata, dtype=np.int16)
    predicao = modelo.predict(audio)

    for nome_modelo, pontuacao in predicao.items():
        if pontuacao > LIMIAR:
            print(f"🔴 ATIVADO! ({nome_modelo}: {pontuacao:.2f})")


with sd.RawInputStream(
    samplerate=TAXA,
    blocksize=TAMANHO_BLOCO,
    device=MICROFONE,
    dtype="int16",
    channels=1,
    callback=callback
):
    while True:
        sd.sleep(100)