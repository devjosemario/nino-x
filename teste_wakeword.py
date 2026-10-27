"""Testa a captura do microfone com um modelo de wake word disponível."""

import numpy as np
import sounddevice as sd
import openwakeword
from openwakeword.model import Model


MODELO_WAKEWORD = "hey_jarvis"
TAXA_AMOSTRAGEM = 16_000
TAMANHO_BLOCO = 1_280  # 80 ms de áudio a 16 kHz
LIMIAR_DETECCAO = 0.5


def main():
    print(f"Preparando o modelo '{MODELO_WAKEWORD}'...")
    openwakeword.utils.download_models(model_names=[MODELO_WAKEWORD])
    modelo = Model(
        wakeword_models=[MODELO_WAKEWORD],
        inference_framework="onnx",
    )
    deteccao_em_andamento = False

    def processar_audio(indata, frames, tempo, status):
        nonlocal deteccao_em_andamento

        if status:
            print(f"Aviso do microfone: {status}")

        audio = np.frombuffer(indata, dtype=np.int16)
        pontuacoes = modelo.predict(audio)
        pontuacao = pontuacoes.get(MODELO_WAKEWORD, 0.0)

        if pontuacao >= LIMIAR_DETECCAO:
            if not deteccao_em_andamento:
                print(f"Detectado! {MODELO_WAKEWORD}: {pontuacao:.2f}", flush=True)
            deteccao_em_andamento = True
        else:
            deteccao_em_andamento = False

    print("Microfone ativo. Diga 'Hey Jarvis'. Pressione Ctrl+C para sair.\n")

    try:
        with sd.RawInputStream(
            samplerate=TAXA_AMOSTRAGEM,
            blocksize=TAMANHO_BLOCO,
            dtype="int16",
            channels=1,
            callback=processar_audio,
        ):
            while True:
                sd.sleep(1_000)
    except KeyboardInterrupt:
        print("\nTeste encerrado.")
    except sd.PortAudioError as erro:
        print(f"Não foi possível abrir o microfone padrão: {erro}")
        print("Confira se o microfone está conectado e selecionado no Windows.")


if __name__ == "__main__":
    main()
