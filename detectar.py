from ultralytics import YOLO
from pathlib import Path
from PIL import Image

MODELO = "modelos/best.pt"
FORMATOS_SUPORTADOS = {'.jpg', '.jpeg', '.png', '.webp', '.bmp', '.mp4', '.avi', '.mov'}

CLASSES = ['Capacete', 'Mascara', 'SEM-Capacete', 'SEM-Mascara', 'SEM-Colete',
           'Pessoa', 'Cone de Seguranca', 'Colete', 'Maquinario', 'Veiculo']

ARQUIVOS = [
    Path("arquivos/construction-safety.jpg"),
    Path("arquivos/portrait-of-woman-with-mask-and-man-with-safety-glasses-on-a-construction-HX01FH.jpg"),
    Path("arquivos/two-young-construction-workers-wearing-555864.jpg"),
    Path("arquivos/download.jfif"),
    Path("arquivos/images.jfif"),
    Path("arquivos/hardhat.mp4"),
]


def preparar_arquivo(caminho: Path) -> Path:
    if caminho.suffix.lower() in FORMATOS_SUPORTADOS:
        return caminho
    destino = Path("saidas/deteccao") / (caminho.stem + ".jpg")
    destino.parent.mkdir(parents=True, exist_ok=True)
    Image.open(caminho).convert("RGB").save(destino)
    print(f"  [convertido {caminho.name} -> {destino.name}]")
    return destino


modelo = YOLO(MODELO)

for arquivo in ARQUIVOS:
    if not arquivo.exists():
        print(f"[IGNORADO] {arquivo} não encontrado")
        continue

    arquivo = preparar_arquivo(arquivo)
    print(f"\n=== {arquivo} ===")
    resultados = modelo(arquivo, conf=0.25)

    for r in resultados:
        if r.boxes is None or len(r.boxes) == 0:
            print("  Nenhuma detecção")
            continue
        for caixa in r.boxes:
            classe = int(caixa.cls[0])
            confianca = float(caixa.conf[0])
            print(f"  {CLASSES[classe]:25s}  conf={confianca:.2f}")

    pasta_saida = Path("saidas/deteccao")
    pasta_saida.mkdir(parents=True, exist_ok=True)
    for r in resultados:
        ext = arquivo.suffix if arquivo.suffix.lower() != '.mp4' else '.jpg'
        r.save(filename=str(pasta_saida / f"{arquivo.stem}_pred{ext}"))

print("\nDetecções salvas em saidas/deteccao/")
