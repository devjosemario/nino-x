import pyttsx3
import webbrowser
import subprocess
from datetime import datetime


voz = pyttsx3.init()

voz.setProperty("rate", 175)
voz.setProperty("volume", 1.0)


def falar(texto):
    print(f"NINO: {texto}")
    voz.say(texto)
    voz.runAndWait()


def executar_comando(comando):

    comando = comando.lower().strip()

    if comando in ["sair", "encerrar", "desligar"]:
        falar("Até logo. Encerrando o sistema.")
        return False

    if "hora" in comando:
        agora = datetime.now()
        falar(f"São {agora.strftime('%H:%M')}.")
        return True

    if "data" in comando:
        agora = datetime.now()
        falar(f"Hoje é {agora.strftime('%d/%m/%Y')}.")
        return True

    if "abrir google" in comando:
        falar("Abrindo o Google.")
        webbrowser.open("https://www.google.com")
        return True

    if "abrir youtube" in comando:
        falar("Abrindo o YouTube.")
        webbrowser.open("https://www.youtube.com")
        return True

    if "abrir github" in comando:
        falar("Abrindo o GitHub.")
        webbrowser.open("https://github.com")
        return True

    if "abrir calculadora" in comando:
        falar("Abrindo a calculadora.")
        subprocess.Popen("calc.exe")
        return True

    if "abrir bloco de notas" in comando:
        falar("Abrindo o bloco de notas.")
        subprocess.Popen("notepad.exe")
        return True

    if "abrir arquivos" in comando:
        falar("Abrindo os arquivos.")
        subprocess.Popen("explorer.exe")
        return True

    if comando.startswith("pesquisar "):
        pesquisa = comando.replace("pesquisar ", "", 1)

        falar(f"Pesquisando por {pesquisa}.")

        url = "https://www.google.com/search?q=" + pesquisa.replace(" ", "+")
        webbrowser.open(url)

        return True

    falar("Ainda não conheço esse comando.")
    return True


falar("Sistemas iniciados.")
falar("Olá. Eu sou NINO.")
falar("Estou pronto para receber seus comandos.")


while True:

    comando = input("\nVocê: ")

    continuar = executar_comando(comando)

    if not continuar:
        break