import shutil
from pathlib import Path
from ultralytics import YOLO

PROJETO = Path(__file__).parent
modelo = YOLO(PROJETO / "modelos/yolov8n.pt")

resultado = modelo.train(
    data=str(PROJETO / "dados/oculos.yaml"),
    epochs=80,
    imgsz=640,
    batch=16,
    patience=15,
    device="mps",
    project=str(PROJETO / "saidas"),
    name="oculos_epi",
    exist_ok=True,
    hsv_h=0.02,
    hsv_s=0.6,
    hsv_v=0.5,
    degrees=15,
    translate=0.1,
    scale=0.5,
    flipud=0.0,
    fliplr=0.5,
    mosaic=1.0,
    mixup=0.1,
    copy_paste=0.1,
)

src = Path(resultado.save_dir) / "weights" / "best.pt"
dst = Path("modelos/oculos.pt")
shutil.copy(src, dst)
print(f"\nTreino concluído. Modelo copiado para {dst}")
