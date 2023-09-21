import torch
import cv2
from PIL import Image
import numpy as np

# Carregar o modelo pré-treinado
model = torch.hub.load('ultralytics/yolov8', 'yolov8s')

# Definir classes dos objetos que o modelo pode detectar
classes = ['person', 'car', 'cat', 'dog']  # Exemplo de classes

# Carregar a imagem
image_path = 'caminho/para/sua/imagem.jpg'  # Altere para o caminho da sua imagem
image = Image.open(image_path).convert('RGB')

# Realizar a detecção de objetos
results = model(image)

# Obter as caixas delimitadoras, classes e probabilidades dos objetos detectados
boxes = results.xyxy[0][:, :4].numpy()
confidences = results.xyxy[0][:, 4].numpy()
class_ids = results.xyxy[0][:, 5].numpy().astype(int)

# Filtrar detecções com confiança acima de um limiar
threshold = 0.5
filtered_indices = np.where(confidences > threshold)[0]
filtered_boxes = boxes[filtered_indices]
filtered_class_ids = class_ids[filtered_indices]

# Mapeamento de IDs de classe para as classes definidas
class_mapping = {0: 'person', 2: 'car', 3: 'cat', 5: 'dog'}

# Desenhar as caixas delimitadoras e rótulos dos objetos detectados
image_np = np.array(image)  # Converter a imagem para um array numpy
image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)  # Converter para formato BGR
for box, class_id in zip(filtered_boxes, filtered_class_ids):
    x1, y1, x2, y2 = map(int, box)  # Converter as coordenadas para inteiros
    class_name = class_mapping.get(class_id, 'Unknown Class')
    cv2.rectangle(image_np, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(image_np, class_name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

# Exibir a imagem com as detecções
cv2.imshow('YOLOv5 Object Detection', image_np)
cv2.waitKey(0)
cv2.destroyAllWindows()
