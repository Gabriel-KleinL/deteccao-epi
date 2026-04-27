"""
Detecção de óculos de proteção EPI vs óculos comum.

Uso:
    python detectar_oculos.py imagem.jpg
    python detectar_oculos.py video.mp4
    python detectar_oculos.py 0          (webcam, índice 0 ou 1)
    python detectar_oculos.py            (webcam padrão)
"""

import sys
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO

MODELO_OCULOS = "saidas/oculos_epi/weights/best.pt"
CONFIANCA = 0.40

CLASSES = {
    0: "Oculos de Protecao",
    1: "Oculos Comum",
}

# Verde para proteção, laranja para comum
CORES = {
    0: (0, 210, 0),
    1: (0, 140, 255),
}


def motivo_classificacao(classe: int, conf: float, bbox) -> str:
    """Gera uma explicação simples da classificação."""
    x1, y1, x2, y2 = bbox
    largura = x2 - x1
    altura = y2 - y1
    proporcao = largura / max(altura, 1)

    if classe == 0:
        # Óculos de proteção tendem a ser mais largos e compactos
        if proporcao > 2.2:
            motivo = "armacao envolvente detectada"
        elif conf > 0.75:
            motivo = "design industrial com protecao lateral"
        else:
            motivo = "formato compativel com EPI"
    else:
        if proporcao > 3.0:
            motivo = "armacao fina e alongada (uso comum)"
        elif conf > 0.75:
            motivo = "sem protecao lateral visivel"
        else:
            motivo = "design nao industrial"

    return motivo


def processar_frame(frame: np.ndarray, modelo: YOLO) -> tuple[np.ndarray, list]:
    resultados_json = []
    deteccoes = modelo(frame, conf=CONFIANCA, verbose=False)

    for r in deteccoes:
        for caixa in r.boxes:
            classe = int(caixa.cls[0])
            conf = float(caixa.conf[0])
            x1, y1, x2, y2 = map(int, caixa.xyxy[0])
            bbox = (x1, y1, x2, y2)

            nome = CLASSES[classe]
            cor = CORES[classe]
            motivo = motivo_classificacao(classe, conf, bbox)

            simbolo = "✓ EPI" if classe == 0 else "⚠ COMUM"
            rotulo = f"{simbolo} {conf:.0%}"

            cv2.rectangle(frame, (x1, y1), (x2, y2), cor, 2)
            (lw, lh), _ = cv2.getTextSize(rotulo, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
            cv2.rectangle(frame, (x1, y1 - lh - 6), (x1 + lw + 4, y1), cor, -1)
            cv2.putText(frame, rotulo, (x1 + 2, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

            resultados_json.append({
                "label": nome,
                "confidence": round(conf, 3),
                "bbox": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
                "motivo": motivo,
            })

    return frame, resultados_json


def rodar_imagem(caminho: Path, modelo: YOLO):
    frame = cv2.imread(str(caminho))
    if frame is None:
        print(f"Erro ao abrir {caminho}")
        return

    frame, resultados = processar_frame(frame, modelo)

    print(f"\n{caminho.name} — {len(resultados)} detecção(ões):")
    for d in resultados:
        print(f"  [{d['label']}]  conf={d['confidence']:.0%}  motivo: {d['motivo']}")

    saida = Path("saidas/deteccao") / f"{caminho.stem}_oculos{caminho.suffix}"
    saida.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(saida), frame)
    print(f"  Salvo em {saida}")


def rodar_camera(fonte, modelo: YOLO):
    cap = cv2.VideoCapture(fonte)
    if not cap.isOpened():
        print(f"Erro: não foi possível abrir '{fonte}'")
        return

    eh_video = isinstance(fonte, str) and Path(fonte).is_file()
    titulo = "Deteccao de Oculos EPI - pressione Q para sair"
    print("Pressione Q para sair.")

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frame, resultados = processar_frame(frame, modelo)

        for d in resultados:
            print(f"  [{d['label']}]  conf={d['confidence']:.0%}  motivo: {d['motivo']}")

        cv2.imshow(titulo, frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


def main():
    if not Path(MODELO_OCULOS).exists():
        print(f"Modelo não encontrado: {MODELO_OCULOS}")
        print("Baixe o dataset e rode primeiro: python treinar_oculos.py")
        return

    modelo = YOLO(MODELO_OCULOS)

    entrada = sys.argv[1] if len(sys.argv) > 1 else "0"

    # Câmera por índice (ex: 0 ou 1)
    if entrada.isdigit():
        rodar_camera(int(entrada), modelo)
        return

    caminho = Path(entrada)
    extensoes_video = {'.mp4', '.avi', '.mov', '.mkv', '.webm'}

    if caminho.suffix.lower() in extensoes_video:
        rodar_camera(str(caminho), modelo)
    else:
        rodar_imagem(caminho, modelo)


if __name__ == "__main__":
    main()
