import cv2
from PIL import Image
import numpy as np
from ultralytics import YOLO
from ultralytics.yolo.v8.detect.predict import DetectionPredictor
import cv2
from helpers import plot_bboxes


def object_detection_video(input_path, output_path, model, threshold=0.5):
    # Abrir o vídeo
    cap = cv2.VideoCapture(input_path)

    # Obter as informações do vídeo (largura, altura, taxa de quadros, etc.)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_path, fourcc, fps,
                          (frame_width, frame_height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Converter o quadro para o formato RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(frame_rgb)
        # Realizar a detecção de objetos
        results = model.predict(image, conf=threshold)

        # Obter as caixas delimitadoras, classes e probabilidades dos objetos detectados
        for detection in results:
            
            plot_bboxes(frame, detection.boxes.data)

        # Escrever o quadro com as detecções no vídeo de saída
        out.write(frame)

    # Liberar os recursos
    cap.release()
    out.release()
    cv2.destroyAllWindows()

    # Liberar os recursos
    cap.release()
    out.release()
    cv2.destroyAllWindows()


# # model = torch.hub.load('ultralytics/yolov8', 'yolov8x')
# model = YOLO("yolov8x.pt")
# # results = model.predict(source="0", show=True) # accepts all formats - img/folder/vid.
# # print(results)
# object_detection_video('videos/video01.mp4', 'video_com_deteccoes.avi', model)
