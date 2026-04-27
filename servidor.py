"""
Ponto de entrada único do sistema EPI.

    python servidor.py          (webcam padrão)
    python servidor.py 1        (outra câmera)

Sobe:
  - Servidor HTTP  → http://localhost:8080
  - Servidor WS    → ws://localhost:8765
  - Abre interface.html e supervisor.html no browser
  - Roda detecção YOLO em loop
"""

import asyncio
import json
import sys
import time
import threading
import webbrowser
import cv2
from functools import partial
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path

from ultralytics import YOLO
import websockets
from websockets.server import serve

# ─── config ──────────────────────────────────────────────────────────────────
MODELO_EPI    = "modelos/best.pt"
MODELO_OCULOS = "modelos/oculos.pt"
HTTP_PORT     = 8080
WS_PORT       = 8765
CONFIANCA     = 0.45
CAMERA_IDX    = int(sys.argv[1]) if len(sys.argv) > 1 else 0

CLASSES_EPI = [
    "Capacete", "Mascara", "SEM-Capacete", "SEM-Mascara",
    "SEM-Colete", "Pessoa", "Cone de Seguranca", "Colete",
    "Maquinario", "Veiculo",
]
CLASSES_OCULOS = ["Oculos de Protecao", "Oculos Comum"]

ALERTAS = {"SEM-Capacete", "SEM-Mascara", "SEM-Colete"}
AVISOS  = {"Oculos Comum"}
SEGUROS = {"Capacete", "Mascara", "Colete", "Oculos de Protecao"}

# ─── clientes WebSocket ───────────────────────────────────────────────────────
clientes: set = set()

async def registrar(websocket):
    clientes.add(websocket)
    print(f"[WS]  Cliente conectado    ({len(clientes)} ativo(s))")
    try:
        await websocket.wait_closed()
    finally:
        clientes.discard(websocket)
        print(f"[WS]  Cliente desconectado ({len(clientes)} ativo(s))")

async def broadcast(evento: dict):
    if not clientes:
        return
    payload = json.dumps(evento, ensure_ascii=False)
    await asyncio.gather(*[c.send(payload) for c in clientes], return_exceptions=True)

# ─── servidor HTTP (serve os arquivos HTML/CSS/JS) ────────────────────────────
class HandlerSilencioso(SimpleHTTPRequestHandler):
    def log_message(self, *_):
        pass  # sem spam no terminal

def iniciar_http():
    servidor = HTTPServer(("localhost", HTTP_PORT), HandlerSilencioso)
    print(f"[HTTP] Servidor em http://localhost:{HTTP_PORT}")
    servidor.serve_forever()

# ─── abrir browser ────────────────────────────────────────────────────────────
def abrir_browser():
    base = f"http://localhost:{HTTP_PORT}"
    time.sleep(1.2)   # aguarda servidores subirem
    webbrowser.open(f"{base}/interface.html")
    time.sleep(0.4)
    webbrowser.open(f"{base}/supervisor.html")

# ─── loop de detecção ─────────────────────────────────────────────────────────
async def loop_deteccao():
    if not Path(MODELO_EPI).exists():
        print(f"[EPI] Modelo não encontrado: {MODELO_EPI}")
        return

    modelo_epi    = YOLO(MODELO_EPI)
    modelo_oculos = YOLO(MODELO_OCULOS) if Path(MODELO_OCULOS).exists() else None

    cap = cv2.VideoCapture(CAMERA_IDX)
    if not cap.isOpened():
        print(f"[CAM] Câmera {CAMERA_IDX} não encontrada.")
        return

    print(f"[CAM] Câmera {CAMERA_IDX} aberta. Iniciando detecção...")

    ultimo_envio:  dict[str, float] = {}
    INTERVALO_MIN    = 3.0
    ultimo_seguro    = 0.0
    INTERVALO_SEGURO = 10.0

    while True:
        ret, frame = cap.read()
        if not ret:
            await asyncio.sleep(0.1)
            continue

        agora = time.time()
        detectados: set[tuple[str, int]] = set()

        for r in modelo_epi(frame, conf=CONFIANCA, verbose=False):
            for caixa in r.boxes:
                cls  = int(caixa.cls[0])
                conf = float(caixa.conf[0])
                detectados.add((CLASSES_EPI[cls], round(conf * 100)))

        if modelo_oculos:
            for r in modelo_oculos(frame, conf=CONFIANCA, verbose=False):
                for caixa in r.boxes:
                    cls  = int(caixa.cls[0])
                    conf = float(caixa.conf[0])
                    detectados.add((CLASSES_OCULOS[cls], round(conf * 100)))

        tem_alerta = False

        for nome, conf in detectados:
            if agora - ultimo_envio.get(nome, 0) < INTERVALO_MIN:
                continue

            if nome in ALERTAS:
                tem_alerta = True
                await broadcast({
                    "tipo":      "alerta",
                    "camera":    f"CAM 0{CAMERA_IDX + 1}",
                    "msg":       f"SEM {nome.replace('SEM-', '')} detectado",
                    "confianca": conf,
                    "zona":      "Dentro",
                })
                ultimo_envio[nome] = agora
                print(f"[ALERTA] {nome}  {conf}%")

            elif nome in AVISOS:
                await broadcast({
                    "tipo":      "aviso",
                    "camera":    f"CAM 0{CAMERA_IDX + 1}",
                    "msg":       f"{nome} detectado",
                    "confianca": conf,
                    "zona":      "Dentro",
                })
                ultimo_envio[nome] = agora
                print(f"[AVISO]  {nome}  {conf}%")

        if not tem_alerta and detectados:
            epi_ok = [n for n, _ in detectados if n in SEGUROS]
            if epi_ok and agora - ultimo_seguro > INTERVALO_SEGURO:
                conf_max = max(c for n, c in detectados if n in SEGUROS)
                await broadcast({
                    "tipo":      "seguro",
                    "camera":    f"CAM 0{CAMERA_IDX + 1}",
                    "msg":       " + ".join(epi_ok[:3]),
                    "confianca": conf_max,
                    "zona":      "Dentro",
                })
                ultimo_seguro = agora
                print(f"[SEGURO] {' + '.join(epi_ok[:3])}")

        await asyncio.sleep(0.05)

    cap.release()

# ─── main ─────────────────────────────────────────────────────────────────────
async def main():
    # HTTP em thread separada
    t_http = threading.Thread(target=iniciar_http, daemon=True)
    t_http.start()

    # Browser em thread separada
    t_browser = threading.Thread(target=abrir_browser, daemon=True)
    t_browser.start()

    print(f"[WS]  Servidor WebSocket em ws://localhost:{WS_PORT}")

    async with serve(registrar, "localhost", WS_PORT):
        await loop_deteccao()

if __name__ == "__main__":
    print("=" * 48)
    print("  Sistema EPI — i9 Automação")
    print("=" * 48)
    asyncio.run(main())
