from ultralytics import YOLO

# Modelo base — yolov8n (leve) ou yolov8s (mais preciso)
modelo = YOLO("modelos/yolov8n.pt")

modelo.train(
    data="dados/oculos.yaml",
    epochs=80,
    imgsz=640,
    batch=16,
    patience=15,           # para cedo se não melhorar
    project="saidas",
    name="oculos_epi",
    # Augmentação para ambientes industriais
    hsv_h=0.02,            # variação de cor (luz artificial vs natural)
    hsv_s=0.6,
    hsv_v=0.5,
    degrees=15,            # rotação leve (câmeras em ângulos variados)
    translate=0.1,
    scale=0.5,             # zoom in/out (distâncias variadas)
    flipud=0.0,
    fliplr=0.5,
    mosaic=1.0,
    mixup=0.1,
    copy_paste=0.1,
)

print("\nTreino concluído. Modelo salvo em saidas/oculos_epi/weights/best.pt")
