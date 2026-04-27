"""
Servidor de detecção EPI com WebSocket.

Roda a câmera + YOLO e envia eventos em tempo real para o supervisor.html.

Uso:
    python servidor.py
    python servidor.py 1       (índice de outra câmera)

Depois abra supervisor.html no navegador.
"""

import asyncio
import json
import sys
import time
import cv2
from pathlib import Path
from ultralytics import YOLO
import websockets
from websockets.server import serve

# ─── config ──────────────────────────────────────────────────────────────────
MODELO_EPI    = "modelos/best.pt"
MODELO_OCULOS = "modelos/oculos.pt"   # opcional
WS_HOST       = "localhost"
WS_PORT       = 8765
CONFIANCA     = 0.45
CAMERA_IDX    = int(sys.argv[1]) if len(sys.argv) > 1 else 0

CLASSES_EPI = [
    "Capacete", "Mascara", "SEM-Capacete", "SEM-Mascara",
    "SEM-Colete", "Pessoa", "Cone de Seguranca", "Colete",
    "Maquinario", "Veiculo",
]
CLASSES_OCULOS = ["Oculos de Protecao", "Oculos Comum"]

# Classes que geram alerta (EPI ausente)
ALERTAS  = {"SEM-Capacete", "SEM-Mascara", "SEM-Colete"}
# Classes que geram aviso
AVISOS   = {"Oculos Comum"}
# Classes que confirmam segurança
SEGUROS  = {"Capacete", "Mascara", "Colete", "Oculos de Protecao"}

# ─── clientes WebSocket conectados ───────────────────────────────────────────
clientes: set = set()

async def registrar(websocket):
    clientes.add(websocket)
    print(f"[WS] Cliente conectado — {len(clientes)} ativo(s)")
    try:
        await websocket.wait_closed()
    finally:
        clientes.discard(websocket)
        print(f"[WS] Cliente desconectado — {len(clientes)} ativo(s)")

async def broadcast(evento: dict):
    if not clientes:
        return
    payload = json.dumps(evento, ensure_ascii=False)
    await asyncio.gather(*[c.send(payload) for c in clientes], return_exceptions=True)

# ─── loop de detecção ────────────────────────────────────────────────────────
async def loop_deteccao():
    if not Path(MODELO_EPI).exists():
        print(f"Modelo não encontrado: {MODELO_EPI}")
        return

    modelo_epi = YOLO(MODELO_EPI)
    modelo_oculos = YOLO(MODELO_OCULOS) if Path(MODELO_OCULOS).exists() else None

    cap = cv2.VideoCapture(CAMERA_IDX)
    if not cap.isOpened():
        print(f"Câmera {CAMERA_IDX} não encontrada.")
        return

    print(f"[CAM] Câmera {CAMERA_IDX} aberta. Detectando...")

    # controle de throttle — evita spam de eventos idênticos
    ultimo_envio: dict[str, float] = {}
    INTERVALO_MIN = 3.0   # segundos entre envios da mesma classe

    ultimo_seguro = 0.0
    INTERVALO_SEGURO = 10.0  # envia "seguro" a cada 10s quando tudo ok

    while True:
        ret, frame = cap.read()
        if not ret:
            await asyncio.sleep(0.1)
            continue

        agora = time.time()
        detectados = set()

        # detecção EPI
        for r in modelo_epi(frame, conf=CONFIANCA, verbose=False):
            for caixa in r.boxes:
                cls  = int(caixa.cls[0])
                conf = float(caixa.conf[0])
                nome = CLASSES_EPI[cls]
                detectados.add((nome, round(conf * 100)))

        # detecção óculos (opcional)
        if modelo_oculos:
            for r in modelo_oculos(frame, conf=CONFIANCA, verbose=False):
                for caixa in r.boxes:
                    cls  = int(caixa.cls[0])
                    conf = float(caixa.conf[0])
                    nome = CLASSES_OCULOS[cls]
                    detectados.add((nome, round(conf * 100)))

        # processa eventos
        tem_alerta = False

        for nome, conf in detectados:
            ultimo = ultimo_envio.get(nome, 0)
            if agora - ultimo < INTERVALO_MIN:
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

            elif nome in AVISOS:
                await broadcast({
                    "tipo":      "aviso",
                    "camera":    f"CAM 0{CAMERA_IDX + 1}",
                    "msg":       f"{nome} detectado",
                    "confianca": conf,
                    "zona":      "Dentro",
                })
                ultimo_envio[nome] = agora

        # envia "seguro" periódico quando só tem EPIs corretos
        if not tem_alerta and detectados:
            epi_ok = [n for n, _ in detectados if n in SEGUROS]
            if epi_ok and agora - ultimo_seguro > INTERVALO_SEGURO:
                await broadcast({
                    "tipo":      "seguro",
                    "camera":    f"CAM 0{CAMERA_IDX + 1}",
                    "msg":       " + ".join(epi_ok[:3]),
                    "confianca": max(c for n, c in detectados if n in SEGUROS),
                    "zona":      "Dentro",
                })
                ultimo_seguro = agora

        await asyncio.sleep(0.05)   # ~20 fps

    cap.release()

# ─── main ─────────────────────────────────────────────────────────────────────
async def main():
    print(f"[WS] Servidor WebSocket em ws://{WS_HOST}:{WS_PORT}")
    print("Abra supervisor.html no navegador.")

    async with serve(registrar, WS_HOST, WS_PORT):
        await loop_deteccao()

if __name__ == "__main__":
    asyncio.run(main())
