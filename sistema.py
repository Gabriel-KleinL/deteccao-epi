"""
Sistema unificado de deteccao de EPI.

Uso:
    python sistema.py              (webcam padrao)
    python sistema.py 1            (outra webcam)
    python sistema.py video.mp4
    python sistema.py imagem.jpg
"""

import sys
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO

MODELO_EPI     = "modelos/best.pt"
MODELO_OCULOS  = "modelos/oculos.pt"   # carregado automaticamente se existir
CONFIANCA      = 0.40

# Classes do modelo principal de EPI
CLASSES_EPI = [
    "Capacete", "Mascara", "SEM-Capacete", "SEM-Mascara",
    "SEM-Colete", "Pessoa", "Cone de Seguranca", "Colete",
    "Maquinario", "Veiculo",
]

# Classes do modelo de oculos
CLASSES_OCULOS = [
    "Oculos de Protecao",
    "Oculos Comum",
]

# Verde = correto, Vermelho = ausente/risco, Neutro = informativo
CORES_EPI = {
    "Capacete":          (0, 200, 0),
    "Mascara":           (0, 200, 0),
    "Colete":            (0, 200, 0),
    "Cone de Seguranca": (0, 200, 255),
    "Pessoa":            (255, 200, 0),
    "SEM-Capacete":      (0, 0, 220),
    "SEM-Mascara":       (0, 0, 220),
    "SEM-Colete":        (0, 0, 220),
    "Maquinario":        (180, 180, 0),
    "Veiculo":           (180, 180, 0),
}

CORES_OCULOS = {
    "Oculos de Protecao": (0, 200, 0),
    "Oculos Comum":       (0, 140, 255),
}


def desenhar_caixa(frame, x1, y1, x2, y2, rotulo, cor):
    cv2.rectangle(frame, (x1, y1), (x2, y2), cor, 2)
    (lw, lh), _ = cv2.getTextSize(rotulo, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
    cv2.rectangle(frame, (x1, y1 - lh - 6), (x1 + lw + 4, y1), cor, -1)
    cv2.putText(frame, rotulo, (x1 + 2, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)


def processar_frame(frame, modelo_epi, modelo_oculos):
    # Deteccao de EPI principal
    for r in modelo_epi(frame, conf=CONFIANCA, verbose=False):
        for caixa in r.boxes:
            cls  = int(caixa.cls[0])
            conf = float(caixa.conf[0])
            nome = CLASSES_EPI[cls]
            cor  = CORES_EPI.get(nome, (200, 200, 200))
            x1, y1, x2, y2 = map(int, caixa.xyxy[0])
            desenhar_caixa(frame, x1, y1, x2, y2, f"{nome} {conf:.0%}", cor)

    # Deteccao de oculos (opcional)
    if modelo_oculos:
        for r in modelo_oculos(frame, conf=CONFIANCA, verbose=False):
            for caixa in r.boxes:
                cls  = int(caixa.cls[0])
                conf = float(caixa.conf[0])
                nome = CLASSES_OCULOS[cls]
                cor  = CORES_OCULOS.get(nome, (200, 200, 200))
                x1, y1, x2, y2 = map(int, caixa.xyxy[0])
                desenhar_caixa(frame, x1, y1, x2, y2, f"{nome} {conf:.0%}", cor)

    return frame


def carregar_modelos():
    if not Path(MODELO_EPI).exists():
        print(f"Erro: modelo principal nao encontrado em {MODELO_EPI}")
        sys.exit(1)

    modelo_epi = YOLO(MODELO_EPI)

    modelo_oculos = None
    if Path(MODELO_OCULOS).exists():
        modelo_oculos = YOLO(MODELO_OCULOS)
        print("Modelo de oculos carregado.")
    else:
        print(f"Modelo de oculos nao encontrado ({MODELO_OCULOS}). Rodando sem deteccao de oculos.")
        print("Para treinar: python treinar_oculos.py")

    return modelo_epi, modelo_oculos


def rodar_camera(fonte, modelo_epi, modelo_oculos):
    cap = cv2.VideoCapture(fonte)
    if not cap.isOpened():
        print(f"Erro: nao foi possivel abrir '{fonte}'")
        return

    print("Pressione Q para sair.")
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = processar_frame(frame, modelo_epi, modelo_oculos)
        cv2.imshow("Sistema de Deteccao de EPI - Q para sair", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


def rodar_imagem(caminho, modelo_epi, modelo_oculos):
    frame = cv2.imread(str(caminho))
    if frame is None:
        print(f"Erro ao abrir {caminho}")
        return
    frame = processar_frame(frame, modelo_epi, modelo_oculos)
    saida = Path("saidas/deteccao") / f"{caminho.stem}_epi{caminho.suffix}"
    saida.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(saida), frame)
    print(f"Salvo em {saida}")


def main():
    modelo_epi, modelo_oculos = carregar_modelos()

    entrada = sys.argv[1] if len(sys.argv) > 1 else "0"

    if entrada.isdigit():
        rodar_camera(int(entrada), modelo_epi, modelo_oculos)
        return

    caminho = Path(entrada)
    if caminho.suffix.lower() in {'.mp4', '.avi', '.mov', '.mkv', '.webm'}:
        rodar_camera(str(caminho), modelo_epi, modelo_oculos)
    else:
        rodar_imagem(caminho, modelo_epi, modelo_oculos)


if __name__ == "__main__":
    main()
