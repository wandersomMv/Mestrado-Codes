from flask import Flask, request, jsonify, send_file
from object_detection import object_detection
import torch
from ultralytics import YOLO
# from prometheus_client import Gauge

from prometheus_flask_exporter import PrometheusMetrics

model = None
app = Flask(__name__)
metrics = PrometheusMetrics(app)

# cpu_usage = Gauge('cpu_usage', 'CPU Usage')

@app.route('/process_image', methods=['POST'])
def process_image():
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400

    image = request.files['image']
    image.save('imagem_recebida.jpg')
    object_detection("imagem_recebida.jpg", model)
    return send_file('imagem_resultante.jpg', mimetype='image/jpeg')


if __name__ == '__main__':
    # Carregar o modelo pré-treinado
    model = YOLO("yolov8x.pt")
    app.run(host='0.0.0.0', port=5000)
   
   
    
    