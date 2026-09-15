import pyttsx3

voz = pyttsx3.init()

voz.setProperty("volume", 1.0)
voz.setProperty("rate", 170)

print("Testando áudio...")

voz.say("Olá. Este é um teste de áudio do JARVIS.")
voz.runAndWait()

print("Teste terminado.")