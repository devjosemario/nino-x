import pyttsx3

voz = pyttsx3.init()
vozes = voz.getProperty("voices")

for i, v in enumerate(vozes):
    print(f"[{i}] {v.name} — {v.id}")
    print(f"    Idiomas: {v.languages}")
    print(f"    Gênero: {v.gender}")
    print()