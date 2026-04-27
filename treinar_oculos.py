"""
Treina o modelo de deteccao de oculos (protecao vs comum).

Antes de rodar:
1. Baixe o dataset no Roboflow Universe buscando por "safety goggles"
   e exporte no formato YOLOv8, colocando as pastas em dados/dataset_oculos/
2. Verifique se dados/oculos.yaml aponta para os caminhos corretos
3. Execute: python treinar_oculos.py

O modelo treinado sera salvo automaticamente em modelos/oculos.pt
e carregado pelo sistema.py junto com o modelo principal.
"""

import shutil
from pathlib import Path
from ultralytics import YOLO

modelo = YOLO("modelos/yolov8n.pt")

modelo.train(
    data="dados/oculos.yaml",
    epochs=80,
    imgsz=640,
    batch=16,
    patience=15,
    project="saidas",
    name="treino_oculos",
    hsv_h=0.02,
    hsv_s=0.6,
    hsv_v=0.5,
    degrees=15,
    translate=0.1,
    scale=0.5,
    fliplr=0.5,
    mosaic=1.0,
    mixup=0.1,
)

# Copia o melhor modelo para a pasta padrao do sistema
origem = Path("saidas/treino_oculos/weights/best.pt")
destino = Path("modelos/oculos.pt")
shutil.copy(origem, destino)
print(f"\nModelo salvo em {destino} — pronto para uso no sistema.py")
