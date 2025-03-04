import threading
import cv2
import os
import time
import numpy as np
from flask import Flask, Response, json, jsonify, request
from flask_cors import CORS
import requests
import websockets
import asyncio
import base64
import queue

app = Flask(__name__)
CORS(app)

TRAINING_IMAGES_PATH = 'Training_images'
capture_done = False
employeeId = -1
frame_queue = queue.Queue(maxsize=20)
capture_queue = queue.Queue(maxsize=20)
socket_running = False
socket_thread = None

def save_images(images, className):
    global socket_running
    person_path = os.path.join(TRAINING_IMAGES_PATH, str(className))
    if not os.path.exists(person_path):
        os.makedirs(person_path)

    for i, img in enumerate(images):
        image_name = f'{className}_{i+1}.jpg'
        cv2.imwrite(os.path.join(person_path, image_name), img)

    socket_running = False
    update_encodings(images,className)

def update_encodings(images, className):
    try:
        images_base64 = [base64.b64encode(cv2.imencode('.jpg', img)[1]).decode('utf-8') for img in images]
        payload = json.dumps({'images': images_base64, 'className': className})

        response = requests.post("http://localhost:8888/reload_encodings", data=payload, headers={'Content-Type': 'application/json'}, timeout=60)
        if response.status_code == 200:
            print("Tải lại encodings thành công.")
        else:
            print(f"Lỗi khi tải lại dữ liệu encodings: status code {response.status_code}")
    except requests.exceptions.RequestException as e:
        print(f"Lỗi kết nối khi tải lại dữ liệu encodings: {e}")
   

def capture_images():
    global capture_done, employeeId
    captured_images = []
    start_time = time.time()
    while len(captured_images) < 10:
        try:
            frame_bytes = capture_queue.get(timeout=1.0)
            img_np = np.frombuffer(frame_bytes, dtype=np.uint8)
            frame = cv2.imdecode(img_np, cv2.IMREAD_COLOR)

            elapsed_time = time.time() - start_time
            if elapsed_time >= 2 and len(captured_images) < 10:
                captured_images.append(frame)
                print(f"Captured image {len(captured_images)}")
                time.sleep(1)
        except queue.Empty:
            continue

    save_images(captured_images, employeeId)
    capture_done = True
    employeeId = -1

async def video_capture_image():
    uri = "ws://192.168.39.137:81"
    try:
        print(f"Connecting to ESP32 CAM at {uri}")
        async with websockets.connect(uri) as websocket:
            print("Connected to ESP32 CAM.")
            while socket_running:
                frame_data = await websocket.recv()
                img_bytes = frame_data
                img_np = np.frombuffer(img_bytes, dtype=np.uint8)
                img = cv2.imdecode(img_np, cv2.IMREAD_COLOR)

                ret, jpeg = cv2.imencode('.jpg', img)
                frame_bytes = jpeg.tobytes()

                try:
                    frame_queue.put(frame_bytes, block=False)
                    capture_queue.put(frame_bytes, block=False)
                except queue.Full:
                    pass

    except websockets.exceptions.ConnectionClosedError:
        print("Connection to ESP32 CAM closed.")
    except Exception as e:
        print(f"Error in video stream: {e}")
    finally:
        print("Video stream ended.")

def generate_video_feed():
    def frame_generator():
        while True:
            try:
                frame_bytes = frame_queue.get(timeout=1.0)
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            except queue.Empty:
                continue

    return frame_generator()

@app.route('/capture_images', methods=['POST'])
def registerId():
    global employeeId
    try:
        data = request.get_json()

        if 'id' not in data:
            return jsonify({'error': 'No ID provided'}), 400

        employeeId = data['id']

        return jsonify({'message': 'ID received successfully', 'employeeId': employeeId}), 200

    except Exception as e:
        return jsonify({'error': f'An error occurred: {str(e)}'}), 500

@app.route('/capture_images', methods=['GET'])
def capture_images_api():
    global capture_done, socket_running, socket_thread
    capture_done = False
    capture_thread = threading.Thread(target=capture_images)
    capture_thread.start()

    if not socket_running:
        socket_running = True
        socket_thread = threading.Thread(target=asyncio.run, args=(video_capture_image(),))
        socket_thread.start()

    return Response(generate_video_feed(), mimetype='multipart/x-mixed-replace; boundary=frame')

def run_flask():
    app.run(host='localhost', port=9999)

if __name__ == "__main__":
    flask_thread = threading.Thread(target=run_flask)
    flask_thread.start()
