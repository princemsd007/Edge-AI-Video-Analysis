from flask import Flask, Response
import cv2
import numpy as np
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing.image import img_to_array
import time
import threading

app = Flask(__name__)

class EdgeAIVideoAnalysis:
    def __init__(self):
        self.video_capture = None
        self.model = None
        self.frame_count = 0
        self.fps = 0
        self.start_time = time.time()
        self.output_frame = None
        self.lock = threading.Lock()

    def initialize_video_capture(self, source=0):
        self.video_capture = cv2.VideoCapture(source)
        if not self.video_capture.isOpened():
            raise ValueError("Unable to open video source")

    def load_ai_model(self):
        self.model = MobileNetV2(weights='imagenet')

    def preprocess_frame(self, frame):
        resized = cv2.resize(frame, (224, 224))
        array = img_to_array(resized)
        expanded = np.expand_dims(array, axis=0)
        preprocessed = preprocess_input(expanded)
        return preprocessed

    def analyze_frame(self, frame):
        preprocessed = self.preprocess_frame(frame)
        predictions = self.model.predict(preprocessed)
        results = decode_predictions(predictions, top=1)[0]
        return results[0]

    def draw_results(self, frame, result):
        label = f"{result[1]}: {result[2]*100:.2f}%"
        cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(frame, f"FPS: {self.fps:.2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    def calculate_fps(self):
        self.frame_count += 1
        elapsed_time = time.time() - self.start_time
        if elapsed_time > 1:
            self.fps = self.frame_count / elapsed_time
            self.frame_count = 0
            self.start_time = time.time()

    def run(self):
        self.initialize_video_capture()
        self.load_ai_model()

        while True:
            ret, frame = self.video_capture.read()
            if not ret:
                break

            result = self.analyze_frame(frame)
            self.draw_results(frame, result)
            self.calculate_fps()

            with self.lock:
                self.output_frame = frame.copy()

    def generate(self):
        while True:
            with self.lock:
                if self.output_frame is None:
                    continue
                (flag, encodedImage) = cv2.imencode(".jpg", self.output_frame)
                if not flag:
                    continue
            yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + 
                  bytearray(encodedImage) + b'\r\n')

edge_ai_system = EdgeAIVideoAnalysis()

@app.route("/")
def index():
    return """
    <html>
      <body>
        <h1>Edge-AI Video Analysis</h1>
        <img src="/video_feed">
      </body>
    </html>
    """

@app.route("/video_feed")
def video_feed():
    return Response(edge_ai_system.generate(),
                    mimetype = "multipart/x-mixed-replace; boundary=frame")

if __name__ == "__main__":
    t = threading.Thread(target=edge_ai_system.run)
    t.daemon = True
    t.start()
    app.run(host='0.0.0.0', port=5000, debug=True, threaded=True, use_reloader=False)