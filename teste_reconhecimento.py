"""Compara Vosk e Faster-Whisper com as mesmas frases gravadas no microfone."""

import json
import time

import numpy as np
import sounddevice as sd
from vosk import KaldiRecognizer, Model


TAXA = 16000
MICROFONE = 1
DURACAO_GRAVACAO = 5
MODELO_VOSK = "vosk-model-small-pt-0.3"
MODELOS_WHISPER = ("base", "small")


def gravar_frase(numero):
    input(f"\nFrase {numero}: pressione Enter e diga um comando em português...")
    print(f"Gravando por {DURACAO_GRAVACAO} segundos...")
    audio = sd.rec(
        int(DURACAO_GRAVACAO * TAXA),
        samplerate=TAXA,
        channels=1,
        dtype="float32",
        device=MICROFONE,
    )
    sd.wait()
    return np.squeeze(audio)


def transcrever_vosk(modelo, audio):
    pcm = np.clip(audio * 32767, -32768, 32767).astype("int16")
    reconhecedor = KaldiRecognizer(modelo, TAXA)
    reconhecedor.AcceptWaveform(pcm.tobytes())
    return json.loads(reconhecedor.FinalResult()).get("text", "")


def transcrever_whisper(modelo, audio):
    segmentos, _ = modelo.transcribe(
        audio,
        language="pt",
        beam_size=1,
        vad_filter=True,
        condition_on_previous_text=False,
    )
    return " ".join(segmento.text.strip() for segmento in segmentos).strip()


def main():
    from faster_whisper import WhisperModel

    print("Carregando Vosk...")
    inicio = time.perf_counter()
    vosk = Model(MODELO_VOSK)
    print(f"Vosk carregado em {time.perf_counter() - inicio:.1f} s")

    whisper = {}
    for tamanho in MODELOS_WHISPER:
        print(f"Carregando Faster-Whisper {tamanho} (CPU, INT8)...")
        inicio = time.perf_counter()
        whisper[tamanho] = WhisperModel(
            tamanho,
            device="cpu",
            compute_type="int8",
            cpu_threads=8,
        )
        print(f"Modelo {tamanho} carregado em {time.perf_counter() - inicio:.1f} s")

    print(
        "\nFale comandos curtos, por exemplo: 'Nino, que horas são?', "
        "'abra o navegador' ou 'o que está na minha tela?'."
    )
    print("Serão gravadas 5 frases; cada áudio será transcrito pelos três modelos.")

    for numero in range(1, 6):
        audio = gravar_frase(numero)
        duracao = len(audio) / TAXA

        inicio = time.perf_counter()
        texto = transcrever_vosk(vosk, audio)
        print(f"Vosk              {time.perf_counter() - inicio:5.2f} s | {texto or '[sem texto]'}")

        for tamanho, modelo in whisper.items():
            inicio = time.perf_counter()
            texto = transcrever_whisper(modelo, audio)
            tempo = time.perf_counter() - inicio
            fator = tempo / duracao if duracao else 0
            print(
                f"Whisper {tamanho:5}     {tempo:5.2f} s "
                f"(RTF {fator:.2f}) | {texto or '[sem texto]'}"
            )

    print("\nTeste concluído. Compare a precisão dos textos e os tempos por frase.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nTeste interrompido.")
