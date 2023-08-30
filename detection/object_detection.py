import torch
import cv2
from PIL import Image
import numpy as np


def object_detection(path, model):


    # Carregar a imagem
    image_path = path  # Altere para o caminho da sua imagem
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
    class_mapping = {0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane', 5: 'bus',
                    6: 'train', 7: 'truck', 8: 'boat', 9: 'traffic light', 10: 'fire hydrant',
                    11: 'stop sign', 12: 'parking meter', 13: 'bench', 14: 'bird', 15: 'cat',
                    16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow', 20: 'elephant', 21: 'bear',
                    22: 'zebra', 23: 'giraffe', 24: 'backpack', 25: 'umbrella', 26: 'handbag',
                    27: 'tie', 28: 'suitcase', 29: 'frisbee', 30: 'skis', 31: 'snowboard',
                    32: 'sports ball', 33: 'kite', 34: 'baseball bat', 35: 'baseball glove',
                    36: 'skateboard', 37: 'surfboard', 38: 'tennis racket', 39: 'bottle',
                    40: 'wine glass', 41: 'cup', 42: 'fork', 43: 'knife', 44: 'spoon',
                    45: 'bowl', 46: 'banana', 47: 'apple', 48: 'sandwich', 49: 'orange',
                    50: 'broccoli', 51: 'carrot', 52: 'hot dog', 53: 'pizza', 54: 'donut',
                    55: 'cake', 56: 'chair', 57: 'couch', 58: 'potted plant', 59: 'bed',
                    60: 'dining table', 61: 'toilet', 62: 'tv', 63: 'laptop', 64: 'mouse',
                    65: 'remote', 66: 'keyboard', 67: 'cell phone', 68: 'microwave',
                    69: 'oven', 70: 'toaster', 71: 'sink', 72: 'refrigerator', 73: 'book',
                    74: 'clock', 75: 'vase', 76: 'scissors', 77: 'teddy bear', 78: 'hair drier',
                    79: 'toothbrush'}

    # Desenhar as caixas delimitadoras e rótulos dos objetos detectados
    image_np = np.array(image)  # Converter a imagem para um array numpy
    image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)  # Converter para formato BGR
    for box, class_id in zip(filtered_boxes, filtered_class_ids):
        x1, y1, x2, y2 = map(int, box)  # Converter as coordenadas para inteiros
        class_name = class_mapping.get(class_id, 'Unknown Class')
        cv2.rectangle(image_np, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(image_np, class_name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Salvar a imagem com as detecções
    cv2.imwrite('imagem_resultante.jpg', image_np)
