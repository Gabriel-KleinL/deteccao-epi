import cv2
from ultralytics import YOLO

MODELO = "modelos/best.pt"
CLASSES = ['Capacete', 'Mascara', 'SEM-Capacete', 'SEM-Mascara', 'SEM-Colete',
           'Pessoa', 'Cone de Seguranca', 'Colete', 'Maquinario', 'Veiculo']

CORES = {
    'Capacete':          (0, 200, 0),
    'Mascara':           (0, 200, 0),
    'Colete':            (0, 200, 0),
    'Cone de Seguranca': (0, 200, 255),
    'Pessoa':            (255, 200, 0),
    'SEM-Capacete':      (0, 0, 220),
    'SEM-Mascara':       (0, 0, 220),
    'SEM-Colete':        (0, 0, 220),
    'Maquinario':        (180, 180, 0),
    'Veiculo':           (180, 180, 0),
}

modelo = YOLO(MODELO)
camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("Erro: câmera não encontrada.")
    exit(1)

print("Pressione Q para sair.")

while True:
    ok, frame = camera.read()
    if not ok:
        break

    resultados = modelo(frame, conf=0.4, verbose=False)

    for r in resultados:
        for caixa in r.boxes:
            classe = int(caixa.cls[0])
            confianca = float(caixa.conf[0])
            nome = CLASSES[classe]
            cor = CORES.get(nome, (200, 200, 200))

            x1, y1, x2, y2 = map(int, caixa.xyxy[0])
            cv2.rectangle(frame, (x1, y1), (x2, y2), cor, 2)

            rotulo = f"{nome} {confianca:.0%}"
            (lw, lh), _ = cv2.getTextSize(rotulo, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
            cv2.rectangle(frame, (x1, y1 - lh - 6), (x1 + lw + 4, y1), cor, -1)
            cv2.putText(frame, rotulo, (x1 + 2, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

    cv2.imshow("Deteccao de EPI - pressione Q para sair", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()
