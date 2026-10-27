"""Complemento Blender: ponte local autenticada para criar uma mesa 3D."""

bl_info = {
    "name": "NinoX Local Bridge",
    "author": "NinoX",
    "version": (1, 0, 0),
    "blender": (3, 6, 0),
    "location": "Preferências > Add-ons",
    "description": "Recebe comandos locais autenticados do NinoX",
    "category": "3D View",
}

import json
import math
import os
import queue
import secrets
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import bpy


HOST = "127.0.0.1"
PORT = 8765
_COMANDOS = queue.Queue()
_SERVIDOR = None
_THREAD_SERVIDOR = None


def _arquivo_token():
    raiz = os.environ.get("LOCALAPPDATA", os.path.expanduser("~"))
    pasta = os.path.join(raiz, "NinoX")
    os.makedirs(pasta, exist_ok=True)
    return os.path.join(pasta, "blender_token.txt")


def _obter_token():
    caminho = _arquivo_token()
    try:
        with open(caminho, "r", encoding="utf-8") as arquivo:
            token = arquivo.read().strip()
    except OSError:
        token = ""
    if not token:
        token = secrets.token_urlsafe(32)
        with open(caminho, "w", encoding="utf-8") as arquivo:
            arquivo.write(token)
    return token


def _responder(handler, status, dados):
    corpo = json.dumps(dados, ensure_ascii=False).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(corpo)))
    handler.end_headers()
    handler.wfile.write(corpo)


class _Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path != "/health":
            _responder(self, 404, {"error": "not_found"})
            return
        if self.headers.get("Authorization") != f"Bearer {_obter_token()}":
            _responder(self, 401, {"error": "unauthorized"})
            return
        _responder(self, 200, {"status": "ready"})

    def do_POST(self):
        if self.path != "/command":
            _responder(self, 404, {"error": "not_found"})
            return
        if self.headers.get("Authorization") != f"Bearer {_obter_token()}":
            _responder(self, 401, {"error": "unauthorized"})
            return
        try:
            tamanho = int(self.headers.get("Content-Length", "0"))
            if tamanho < 1 or tamanho > 4096:
                raise ValueError("invalid_size")
            comando = json.loads(self.rfile.read(tamanho).decode("utf-8"))
            if not isinstance(comando, dict) or comando.get("action") != "create_table":
                raise ValueError("unsupported_action")
        except (ValueError, UnicodeDecodeError, json.JSONDecodeError):
            _responder(self, 400, {"error": "invalid_command"})
            return

        tarefa = {"command": comando, "ready": threading.Event(), "result": {}}
        _COMANDOS.put(tarefa)
        if not tarefa["ready"].wait(timeout=12):
            _responder(self, 504, {"error": "blender_busy"})
            return
        status = 200 if "error" not in tarefa["result"] else 400
        _responder(self, status, tarefa["result"])

    def log_message(self, _format, *_args):
        return


def _numero(comando, chave, padrao, minimo, maximo):
    valor = float(comando.get(chave, padrao))
    if not math.isfinite(valor) or not minimo <= valor <= maximo:
        raise ValueError(f"Medida inválida: {chave}")
    return valor


def _criar_prisma(nome, dimensoes, localizacao, material):
    largura, profundidade, altura = dimensoes
    x, y, z = largura / 2, profundidade / 2, altura / 2
    vertices = [
        (-x, -y, -z), (x, -y, -z), (x, y, -z), (-x, y, -z),
        (-x, -y, z), (x, -y, z), (x, y, z), (-x, y, z),
    ]
    faces = [
        (0, 3, 2, 1), (4, 5, 6, 7), (0, 1, 5, 4),
        (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7),
    ]
    malha = bpy.data.meshes.new(f"{nome} Mesh")
    malha.from_pydata(vertices, [], faces)
    malha.materials.append(material)
    objeto = bpy.data.objects.new(nome, malha)
    objeto.location = localizacao
    return objeto


def _criar_cilindro(nome, raio, altura, localizacao, material, lados=32):
    vertices = []
    for z in (-altura / 2, altura / 2):
        vertices.extend(
            (raio * math.cos(2 * math.pi * i / lados),
             raio * math.sin(2 * math.pi * i / lados), z)
            for i in range(lados)
        )
    faces = [tuple(reversed(range(lados))), tuple(range(lados, lados * 2))]
    faces.extend(
        (i, (i + 1) % lados, (i + 1) % lados + lados, i + lados)
        for i in range(lados)
    )
    malha = bpy.data.meshes.new(f"{nome} Mesh")
    malha.from_pydata(vertices, [], faces)
    malha.materials.append(material)
    objeto = bpy.data.objects.new(nome, malha)
    objeto.location = localizacao
    return objeto


def _criar_disco(nome, diametro, altura, localizacao, material, lados=64):
    raio = diametro / 2
    vertices = []
    for z in (-altura / 2, altura / 2):
        vertices.extend(
            (raio * math.cos(2 * math.pi * i / lados),
             raio * math.sin(2 * math.pi * i / lados), z)
            for i in range(lados)
        )
    faces = [tuple(reversed(range(lados))), tuple(range(lados, lados * 2))]
    faces.extend(
        (i, (i + 1) % lados, (i + 1) % lados + lados, i + lados)
        for i in range(lados)
    )
    malha = bpy.data.meshes.new(f"{nome} Mesh")
    malha.from_pydata(vertices, [], faces)
    malha.materials.append(material)
    objeto = bpy.data.objects.new(nome, malha)
    objeto.location = localizacao
    return objeto


def _material(nome, cor):
    material = bpy.data.materials.new(nome)
    material.diffuse_color = (*cor, 1.0)
    material.use_nodes = True
    shader = material.node_tree.nodes.get("Principled BSDF")
    if shader:
        shader.inputs["Base Color"].default_value = (*cor, 1.0)
        shader.inputs["Roughness"].default_value = 0.42
    return material


def _criar_mesa(comando):
    diametro = comando.get("diameter")
    redonda = diametro is not None
    if redonda:
        diametro = _numero(comando, "diameter", 1.0, 0.2, 10.0)
        largura = profundidade = diametro
    else:
        largura = _numero(comando, "width", 1.2, 0.2, 10.0)
        profundidade = _numero(comando, "depth", 0.75, 0.2, 10.0)
    altura = _numero(comando, "height", 0.75, 0.25, 5.0)
    espessura = _numero(comando, "thickness", 0.04, 0.01, 0.5)
    if espessura >= altura:
        raise ValueError("A espessura precisa ser menor que a altura da mesa.")

    nome = str(comando.get("name") or "Mesa do Nino")[:80]
    colecao = bpy.data.collections.get("NinoX")
    if colecao is None:
        colecao = bpy.data.collections.new("NinoX")
        bpy.context.scene.collection.children.link(colecao)

    material_tampo = _material("NinoX Madeira", (0.30, 0.12, 0.045))
    material_pes = _material("NinoX Estrutura", (0.12, 0.12, 0.14))
    objetos = []

    if redonda:
        tampo = _criar_disco(
            f"{nome} - Tampo", diametro, espessura,
            (0, 0, altura - espessura / 2), material_tampo,
        )
    else:
        tampo = _criar_prisma(
            f"{nome} - Tampo",
            (largura, profundidade, espessura),
            (0, 0, altura - espessura / 2),
            material_tampo,
        )
    objetos.append(tampo)

    altura_pes = altura - espessura
    raio = min(largura, profundidade) * 0.035
    if redonda:
        recuo_x = recuo_y = diametro * 0.28
    else:
        recuo_x = min(largura * 0.12, max(raio * 2, 0.04))
        recuo_y = min(profundidade * 0.12, max(raio * 2, 0.04))
    for x in (-largura / 2 + recuo_x, largura / 2 - recuo_x):
        for y in (-profundidade / 2 + recuo_y, profundidade / 2 - recuo_y):
            objetos.append(
                _criar_cilindro(
                    f"{nome} - Pé",
                    raio,
                    altura_pes,
                    (x, y, altura_pes / 2),
                    material_pes,
                )
            )

    for objeto in objetos:
        colecao.objects.link(objeto)
    for existente in bpy.context.scene.objects:
        existente.select_set(False)
    for objeto in objetos:
        objeto.select_set(True)
    bpy.context.view_layer.objects.active = tampo
    return {"status": "created", "objects": len(objetos), "name": nome}


def _processar_comandos():
    while True:
        try:
            tarefa = _COMANDOS.get_nowait()
        except queue.Empty:
            break
        try:
            tarefa["result"] = _criar_mesa(tarefa["command"])
        except Exception as erro:
            tarefa["result"] = {"error": str(erro)}
        finally:
            tarefa["ready"].set()
    return 0.1


def _iniciar_servidor():
    global _SERVIDOR, _THREAD_SERVIDOR
    if _SERVIDOR is not None:
        return
    _obter_token()
    try:
        _SERVIDOR = ThreadingHTTPServer((HOST, PORT), _Handler)
    except OSError as erro:
        print(f"NinoX Blender Bridge: não foi possível abrir {HOST}:{PORT}: {erro}")
        return
    _THREAD_SERVIDOR = threading.Thread(target=_SERVIDOR.serve_forever, daemon=True)
    _THREAD_SERVIDOR.start()
    bpy.app.timers.register(_processar_comandos, first_interval=0.1)
    print(f"NinoX Blender Bridge ativo em http://{HOST}:{PORT}")


def _parar_servidor():
    global _SERVIDOR, _THREAD_SERVIDOR
    if _SERVIDOR is not None:
        _SERVIDOR.shutdown()
        _SERVIDOR.server_close()
        _SERVIDOR = None
        _THREAD_SERVIDOR = None
    if bpy.app.timers.is_registered(_processar_comandos):
        bpy.app.timers.unregister(_processar_comandos)


def register():
    _iniciar_servidor()


def unregister():
    _parar_servidor()


if __name__ == "__main__":
    register()
