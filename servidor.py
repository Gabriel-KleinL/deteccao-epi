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
import atexit
import json
import os
import signal
import socket
import sys
import time
import threading
import webbrowser

os.environ.setdefault("OPENCV_LOG_LEVEL", "SILENT")
os.environ.setdefault("OBSENSOR_LOG_LEVEL", "off")
import platform
import cv2

# No macOS força AVFoundation para evitar o plugin Orbbec (que spama logs)
_CAM_BACKEND = cv2.CAP_AVFOUNDATION if platform.system() == "Darwin" else cv2.CAP_ANY


def _abrir_camera(idx: int) -> cv2.VideoCapture:
    """Abre câmera com backend nativo, suprimindo logs de SDKs de terceiros."""
    devnull_fd = os.open(os.devnull, os.O_WRONLY)
    old_stderr  = os.dup(2)
    try:
        os.dup2(devnull_fd, 2)
        cap = cv2.VideoCapture(idx, _CAM_BACKEND)
        time.sleep(0.2)   # aguarda threads do SDK terminarem de logar
    finally:
        os.dup2(old_stderr, 2)
        os.close(old_stderr)
        os.close(devnull_fd)
    return cap

from datetime import datetime, timezone
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path

BASE = Path(__file__).parent

from ultralytics import YOLO
import websockets

# ─── carregar config.json ─────────────────────────────────────────────────────
_cfg: dict = {}
_cfg_path = BASE / "config.json"
if _cfg_path.exists():
    try:
        _cfg = json.loads(_cfg_path.read_text(encoding="utf-8"))
    except Exception:
        print("[CFG] config.json inválido — usando padrões.")

def _lan_ip() -> str:
    """Retorna o IP da máquina na rede local."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"

# câmera global para liberação no shutdown
_cap_global = None

# buffer MJPEG para streaming HTTP (acesso thread-safe)
_ultimo_frame_jpg: bytes = b""
_frame_lock = threading.Lock()

# ─── zona ────────────────────────────────────────────────────────────────────
zonas: list[dict] = []


def _ponto_dentro(px: float, py: float, pontos: list) -> bool:
    n = len(pontos)
    dentro = False
    j = n - 1
    for i in range(n):
        xi, yi = pontos[i]["x"], pontos[i]["y"]
        xj, yj = pontos[j]["x"], pontos[j]["y"]
        if ((yi > py) != (yj > py)) and (px < (xj - xi) * (py - yi) / (yj - yi) + xi):
            dentro = not dentro
        j = i
    return dentro


def _em_zona(bbox: list, zona: dict) -> bool:
    pontos = zona.get("pontos", [])
    if not pontos:
        return False
    cx = (bbox[0] + bbox[2]) / 2
    cy = (bbox[1] + bbox[3]) / 2
    dentro = _ponto_dentro(cx, cy, pontos)
    return dentro if zona.get("modo", "dentro") == "dentro" else not dentro


def zonas_da_deteccao(bbox: list, nome: str) -> list[dict]:
    return [
        z for z in zonas
        if z.get("ativa", True)
        and nome in z.get("classes", [])
        and _em_zona(bbox, z)
    ]


def bbox_visivel(bbox: list) -> bool:
    ativas = [z for z in zonas if z.get("ativa", True)]
    if not ativas:
        return True
    return any(_em_zona(bbox, z) for z in ativas)

# ─── config ──────────────────────────────────────────────────────────────────
MODELO_EPI       = BASE / "modelos/best.pt"
MODELO_OCULOS    = BASE / "modelos/oculos.pt"
HTTP_PORT        = int(_cfg.get("http_port",   8080))
WS_PORT          = int(_cfg.get("ws_port",     8765))
CONFIANCA        = float(_cfg.get("confianca",        0.45))
CONFIANCA_ALERTA = float(_cfg.get("confianca_alerta", 0.70))
CAMERA_IDX       = int(sys.argv[1]) if len(sys.argv) > 1 else int(_cfg.get("camera", 0))
_LOG_DIAS        = int(_cfg.get("log_manter_dias",      30))
_CAPTURAS_DIAS   = int(_cfg.get("capturas_manter_dias",  7))

CLASSES_EPI = [
    "Capacete", "Mascara", "SEM-Capacete", "SEM-Mascara",
    "SEM-Colete", "Pessoa", "Cone de Seguranca", "Colete",
    "Maquinario", "Veiculo",
]
CLASSES_OCULOS = ["Oculos de Protecao", "SEM-Oculos"]

ALERTAS = {"SEM-Capacete", "SEM-Mascara", "SEM-Colete", "SEM-Oculos"}
AVISOS  = set()
SEGUROS = {"Capacete", "Mascara", "Colete", "Oculos de Protecao"}

# ─── clientes WebSocket ───────────────────────────────────────────────────────
clientes: set = set()
classes_habilitadas: set = set()
intervalo_min: float = float(_cfg.get("intervalo_min", 3.0))


async def registrar(websocket):
    clientes.add(websocket)
    print(f"[WS]  Cliente conectado    ({len(clientes)} ativo(s))")
    try:
        async for msg in websocket:
            try:
                ev = json.loads(msg)

                if ev.get("tipo") == "zonas":
                    zonas.clear()
                    zonas.extend(ev.get("zonas", []))
                    print(f"[ZONA] {len(zonas)} zona(s) ativa(s)")

                elif ev.get("tipo") == "config":
                    classes_habilitadas.clear()
                    classes_habilitadas.update(ev.get("habilitados", []))
                    print(f"[CFG]  {len(classes_habilitadas)} classe(s): "
                          f"{', '.join(sorted(classes_habilitadas)) or 'nenhuma'}")

                elif ev.get("tipo") == "intervalo":
                    global intervalo_min
                    intervalo_min = float(ev.get("segundos", 3))
                    print(f"[CFG]  Intervalo de alertas: {intervalo_min}s")

                elif ev.get("tipo") == "confianca":
                    global CONFIANCA, CONFIANCA_ALERTA
                    CONFIANCA        = float(ev.get("conf",        CONFIANCA))
                    CONFIANCA_ALERTA = float(ev.get("conf_alerta", CONFIANCA_ALERTA))
                    print(f"[CFG]  Confiança: geral={CONFIANCA:.0%}  alerta={CONFIANCA_ALERTA:.0%}")
            except Exception:
                pass
    finally:
        clientes.discard(websocket)
        print(f"[WS]  Cliente desconectado ({len(clientes)} ativo(s))")


async def broadcast(evento: dict):
    if not clientes:
        return
    payload = json.dumps(evento, ensure_ascii=False)
    await asyncio.gather(*[c.send(payload) for c in clientes], return_exceptions=True)


def salvar_log(evento: dict):
    """Persiste o evento em resultados/log_YYYY-MM-DD.jsonl (JSON Lines)."""
    registro = dict(evento)
    registro["ts"] = datetime.now(timezone.utc).isoformat()
    dir_resultados = BASE / "resultados"
    dir_resultados.mkdir(exist_ok=True)
    nome_arquivo = dir_resultados / f"log_{datetime.now().strftime('%Y-%m-%d')}.jsonl"
    with nome_arquivo.open("a", encoding="utf-8") as f:
        f.write(json.dumps(registro, ensure_ascii=False) + "\n")


def limpar_logs_antigos(manter_dias: int = 30):
    """Remove arquivos de log com mais de `manter_dias` dias."""
    dir_resultados = BASE / "resultados"
    if not dir_resultados.exists():
        return
    cutoff = time.time() - manter_dias * 86400
    for f in dir_resultados.glob("log_*.jsonl"):
        if f.stat().st_mtime < cutoff:
            try:
                f.unlink()
                print(f"[LOG] Removido log antigo: {f.name}")
            except Exception:
                pass


def limpar_capturas_antigas(manter_dias: int = 7):
    """Remove capturas de alerta com mais de `manter_dias` dias."""
    dir_capturas = BASE / "resultados" / "capturas"
    if not dir_capturas.exists():
        return
    cutoff = time.time() - manter_dias * 86400
    removidos = 0
    for f in dir_capturas.glob("*.jpg"):
        if f.stat().st_mtime < cutoff:
            try:
                f.unlink()
                removidos += 1
            except Exception:
                pass
    if removidos:
        print(f"[LOG] Removidas {removidos} captura(s) antiga(s).")

# ─── servidor HTTP ────────────────────────────────────────────────────────────
class HandlerSilencioso(SimpleHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/api/config':
            corpo = json.dumps({"ws_port": WS_PORT, "http_port": HTTP_PORT}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", len(corpo))
            self.end_headers()
            self.wfile.write(corpo)
        elif self.path == '/api/capturas':
            pasta = BASE / "resultados" / "capturas"
            arquivos = sorted(pasta.glob("*.jpg"), key=lambda p: p.stat().st_mtime, reverse=True)[:30] if pasta.exists() else []
            corpo = json.dumps([p.name for p in arquivos]).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", len(corpo))
            self.end_headers()
            self.wfile.write(corpo)
        elif self.path == '/stream':
            self.send_response(200)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=--jpgboundary")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            try:
                while True:
                    with _frame_lock:
                        jpg = _ultimo_frame_jpg
                    if jpg:
                        header = (
                            b"--jpgboundary\r\n"
                            b"Content-Type: image/jpeg\r\n"
                            + f"Content-Length: {len(jpg)}\r\n\r\n".encode()
                        )
                        self.wfile.write(header + jpg + b"\r\n")
                        self.wfile.flush()
                    time.sleep(0.1)
            except Exception:
                pass
        else:
            super().do_GET()
    def log_message(self, *_):
        pass


def iniciar_http():
    os.chdir(BASE)
    servidor = HTTPServer(("0.0.0.0", HTTP_PORT), HandlerSilencioso)
    ip = _lan_ip()
    print(f"[HTTP] Local:  http://localhost:{HTTP_PORT}")
    print(f"[HTTP] Rede:   http://{ip}:{HTTP_PORT}")
    servidor.serve_forever()

# ─── abrir browser ────────────────────────────────────────────────────────────
def abrir_browser():
    base = f"http://localhost:{HTTP_PORT}"
    time.sleep(1.2)
    webbrowser.open(f"{base}/interface.html")
    time.sleep(0.4)
    webbrowser.open(f"{base}/supervisor.html")

# ─── loop de detecção ─────────────────────────────────────────────────────────
async def loop_deteccao():
    global _cap_global

    if not Path(MODELO_EPI).exists():
        print(f"[EPI] Modelo não encontrado: {MODELO_EPI}")
        return

    modelo_epi    = YOLO(MODELO_EPI)
    modelo_oculos = YOLO(MODELO_OCULOS) if Path(MODELO_OCULOS).exists() else None

    cap = None
    while True:
        cap = _abrir_camera(CAMERA_IDX)
        if cap.isOpened():
            break
        print(f"[CAM] Câmera {CAMERA_IDX} não encontrada. Tentando novamente em 5s...")
        cap.release()
        await asyncio.sleep(5)

    _cap_global = cap
    print(f"[CAM] Câmera {CAMERA_IDX} aberta. Iniciando detecção...")

    ultimo_envio:  dict[tuple, float] = {}  # chave: (nome, zona_id)
    ultimo_seguro    = 0.0
    INTERVALO_SEGURO = 10.0
    ultimo_frame_ws  = 0.0
    INTERVALO_FRAME  = 0.1
    falhas_consecutivas = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            falhas_consecutivas += 1
            await asyncio.sleep(0.1)
            if falhas_consecutivas >= 30:
                cap.release()
                print("[CAM] Câmera perdida. Tentando reconectar...")
                while True:
                    cap = _abrir_camera(CAMERA_IDX)
                    if cap.isOpened():
                        break
                    print(f"[CAM] Câmera {CAMERA_IDX} não encontrada. Tentando novamente em 5s...")
                    cap.release()
                    await asyncio.sleep(5)
                _cap_global = cap
                falhas_consecutivas = 0
                print(f"[CAM] Câmera {CAMERA_IDX} reconectada.")
            continue
        falhas_consecutivas = 0

        # atualiza buffer MJPEG para /stream
        ok, _jpg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
        if ok:
            with _frame_lock:
                _ultimo_frame_jpg = _jpg.tobytes()

        agora = time.time()
        caixas_frame: list[dict] = []
        # detectados: list of (nome, conf, zona_nome, zona_id, zona_cor)
        detectados: list[tuple] = []

        def processar_deteccao(nome, conf_raw, bbox):
            limiar = CONFIANCA_ALERTA if nome in ALERTAS else CONFIANCA
            if conf_raw < limiar:
                return
            conf = round(conf_raw * 100)
            caixas_frame.append({"nome": nome, "conf": conf, "bbox": bbox})

            if zonas:
                # zonas definem quais classes geram eventos — toggles individuais não interferem
                for z in zonas_da_deteccao(bbox, nome):
                    detectados.append((nome, conf, z["nome"], z["id"], z.get("cor", "#C73C3C")))
            else:
                # sem zonas: toggles individuais são a autoridade
                if nome not in classes_habilitadas:
                    return
                detectados.append((nome, conf, "Geral", "__geral__", "#C73C3C"))

        try:
            for r in modelo_epi(frame, conf=CONFIANCA, verbose=False):
                for caixa in r.boxes:
                    cls  = int(caixa.cls[0])
                    conf = float(caixa.conf[0])
                    nome = CLASSES_EPI[cls]
                    x1, y1, x2, y2 = caixa.xyxyn[0].tolist()
                    bbox = [round(x1, 4), round(y1, 4), round(x2, 4), round(y2, 4)]
                    processar_deteccao(nome, conf, bbox)
        except Exception as e:
            print(f"[YOLO] Erro EPI: {e}")
            await asyncio.sleep(0.5)

        if modelo_oculos:
            try:
                for r in modelo_oculos(frame, conf=CONFIANCA, verbose=False):
                    for caixa in r.boxes:
                        cls  = int(caixa.cls[0])
                        conf = float(caixa.conf[0])
                        nome = CLASSES_OCULOS[cls]
                        x1, y1, x2, y2 = caixa.xyxyn[0].tolist()
                        bbox = [round(x1, 4), round(y1, 4), round(x2, 4), round(y2, 4)]
                        processar_deteccao(nome, conf, bbox)
            except Exception as e:
                print(f"[YOLO] Erro óculos: {e}")

        # envia caixas filtradas por zona ao browser
        if agora - ultimo_frame_ws >= INTERVALO_FRAME:
            caixas_visiveis = [c for c in caixas_frame if bbox_visivel(c["bbox"])]
            pessoas = sum(1 for c in caixas_frame if c["nome"] == "Pessoa")
            await broadcast({"tipo": "frame", "caixas": caixas_visiveis, "pessoas": pessoas})
            ultimo_frame_ws = agora

        tem_alerta = False

        for nome, conf, zona_nome, zona_id, zona_cor in detectados:
            chave = (nome, zona_id)
            if agora - ultimo_envio.get(chave, 0) < intervalo_min:
                continue

            if nome in ALERTAS:
                tem_alerta = True
                ev_alerta = {
                    "tipo":      "alerta",
                    "camera":    f"CAM 0{CAMERA_IDX + 1}",
                    "msg":       f"SEM {nome.replace('SEM-', '')} detectado",
                    "confianca": conf,
                    "zona":      zona_nome,
                    "cor_zona":  zona_cor,
                }
                await broadcast(ev_alerta)
                salvar_log(ev_alerta)
                ultimo_envio[chave] = agora
                print(f"[ALERTA] {nome}  {conf}%  [{zona_nome}]")

                ts_captura = datetime.now().strftime("%Y%m%d_%H%M%S")
                nome_arquivo = f"{ts_captura}_{nome}.jpg"
                dir_capturas = BASE / "resultados/capturas"
                dir_capturas.mkdir(parents=True, exist_ok=True)
                caminho_captura = dir_capturas / nome_arquivo
                threading.Thread(
                    target=cv2.imwrite,
                    args=(str(caminho_captura), frame.copy()),
                    daemon=True,
                ).start()
                print(f"[CAPTURA] {nome} → {caminho_captura}")

            elif nome in AVISOS:
                ev_aviso = {
                    "tipo":      "aviso",
                    "camera":    f"CAM 0{CAMERA_IDX + 1}",
                    "msg":       f"{nome} detectado",
                    "confianca": conf,
                    "zona":      zona_nome,
                    "cor_zona":  zona_cor,
                }
                await broadcast(ev_aviso)
                salvar_log(ev_aviso)
                ultimo_envio[chave] = agora
                print(f"[AVISO]  {nome}  {conf}%  [{zona_nome}]")

        if not tem_alerta and detectados:
            epi_ok = list({n for n, *_ in detectados if n in SEGUROS})
            if epi_ok and agora - ultimo_seguro > INTERVALO_SEGURO:
                conf_max = max(c for n, c, *_ in detectados if n in SEGUROS)
                ev_seguro = {
                    "tipo":      "seguro",
                    "camera":    f"CAM 0{CAMERA_IDX + 1}",
                    "msg":       " + ".join(epi_ok[:3]),
                    "confianca": conf_max,
                    "zona":      "Geral",
                }
                await broadcast(ev_seguro)
                salvar_log(ev_seguro)
                ultimo_seguro = agora
                print(f"[SEGURO] {' + '.join(epi_ok[:3])}")

        await asyncio.sleep(0.05)

    cap.release()

# ─── main ─────────────────────────────────────────────────────────────────────
async def main():
    t_http    = threading.Thread(target=iniciar_http, daemon=True)
    t_browser = threading.Thread(target=abrir_browser, daemon=True)
    t_http.start()
    t_browser.start()

    ip = _lan_ip()
    print(f"[WS]  Local:  ws://localhost:{WS_PORT}")
    print(f"[WS]  Rede:   ws://{ip}:{WS_PORT}")

    async with websockets.serve(registrar, "0.0.0.0", WS_PORT):
        await loop_deteccao()

_encerrado = False

def _liberar_camera():
    global _encerrado
    if _encerrado:
        return
    _encerrado = True
    if _cap_global is not None and _cap_global.isOpened():
        _cap_global.release()
        print("[SYS] Câmera liberada.")

atexit.register(_liberar_camera)

def _sigterm(sig, frame):
    print("\n[SYS] Sinal SIGTERM recebido.")
    sys.exit(0)

signal.signal(signal.SIGTERM, _sigterm)

if __name__ == "__main__":
    print("=" * 48)
    print("  Sistema EPI — i9 Automação")
    print("=" * 48)
    limpar_logs_antigos(_LOG_DIAS)
    limpar_capturas_antigas(_CAPTURAS_DIAS)
    try:
        asyncio.run(main())
    except (KeyboardInterrupt, SystemExit):
        print("\n[SYS] Servidor encerrado pelo usuário.")
