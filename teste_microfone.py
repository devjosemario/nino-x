import sounddevice as sd
import numpy as np

DURACAO = 5
TAXA = 44100

print("🎙️ Fale alguma coisa por 5 segundos...")

audio = sd.rec(
    int(DURACAO * TAXA),
    samplerate=TAXA,
    channels=1,
    dtype="float32",
    device=1
)

sd.wait()

volume = np.max(np.abs(audio))

print(f"Volume captado: {volume:.4f}")

if volume > 0.01:
    print("✅ Microfone funcionando!")
else:
    print("⚠️ Não detectei som. Verifique o microfone.")