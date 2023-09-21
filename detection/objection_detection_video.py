import torch
import cv2
from PIL import Image
import numpy as np

# Carregar o modelo pré-treinado
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

# Definir classes dos objetos que o modelo pode detectar
classes = ['person', 'car', 'cat', 'dog']  # Exemplo de classes

# Configuração da câmera
cap = cv2.VideoCapture(1)  # Usar webcam (ou alterar para o caminho do vídeo)

while True:
    ret, frame = cap.read()  # Capturar o quadro da câmera

    # Converter o quadro para o formato adequado (RGB)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_pil = Image.fromarray(frame_rgb)

    # Realizar a detecção de objetos
    results = model(frame_pil)

    # Obter as caixas delimitadoras, classes e probabilidades dos objetos detectados
    boxes = results.xyxy[0].numpy()
    confidences = results.xyxys[0, :, 4].numpy()
    class_ids = results.xyxys[0, :, 5].numpy().astype(int)

    # Filtrar detecções com confiança acima de um limiar
    threshold = 0.5
    filtered_indices = np.where(confidences > threshold)[0]
    filtered_boxes = boxes[filtered_indices]
    filtered_class_ids = class_ids[filtered_indices]

    # Desenhar as caixas delimitadoras e rótulos dos objetos detectados
    for box, class_id in zip(filtered_boxes, filtered_class_ids):
        x1, y1, x2, y2 = box
        class_name = classes[class_id]
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, class_name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Exibir o quadro com as detecções
    cv2.imshow('YOLOv5 Object Detection', frame)

    # Parar o loop se a tecla 'q' for pressionada
    if cv2.waitKey(1) == ord('q'):
        break

# Liberar recursos
cap.release()
cv2.destroyAllWindows()
