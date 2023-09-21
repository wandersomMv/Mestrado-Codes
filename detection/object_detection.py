import torch
import cv2
from PIL import Image
import numpy as np
from helpers import plot_bboxes

def object_detection(path, model, threshold=0.5):


    # Carregar a imagem
    image_path = path  # Altere para o caminho da sua imagem
    image = cv2.imread(image_path)
    img_predict = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img_predict = Image.fromarray(img_predict)
    

    # Realizar a detecção de objetos
    results = model.predict(img_predict, conf=threshold)
            # Obter as caixas delimitadoras, classes e probabilidades dos objetos detectados
    for detection in results:
            
            plot_bboxes(image, detection.boxes.data)
    # Obter as caixas delimitadoras, classes e probabilidades dos objetos detectados
    

    # Salvar a imagem com as detecções
    cv2.imwrite('imagem_resultante.jpg', image)
