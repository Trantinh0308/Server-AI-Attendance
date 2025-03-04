import base64
import os
import threading
import time
import cv2
import face_recognition
import numpy as np
from flask import Flask, Response, jsonify, request
from flask_cors import CORS
import websockets
import asyncio
import queue
from utils import findEncodings, load_training_images, send_attendance_request, send_control_device
import datetime

app = Flask(__name__)
CORS(app)

TRAINING_IMAGES_PATH = 'Training_images'
camera_lock = threading.Lock()
last_name = ""
last_request_time = 0
encodeListKnown = []
classNames = []
frame_queue = queue.Queue(maxsize=10)
socket_running = False
socket_thread = None
servo_open = False


def init_encoding():
    global encodeListKnown, classNames
    with camera_lock:
        images, classNames = load_training_images(TRAINING_IMAGES_PATH)
        encodeListKnown = findEncodings(images)
        print("Tải encodings thành công.")

init_encoding()

def handle_request_with_delay(name, img, current_time):
    global last_name, last_request_time
    current_time_in_seconds = time.time()
    if name != last_name or (current_time_in_seconds - last_request_time) >= 600:
        print(f"Sending attendance request for {name} at {current_time.strftime('%H:%M:%S')}")
        send_attendance_request(name, img, current_time)
        last_name = name
        last_request_time = current_time_in_seconds


servo_queue = queue.Queue()
attendance_queue = queue.Queue()

def process_servo_queue():
    """Xử lý hàng đợi mở/đóng cửa servo."""
    global servo_open
    while True:
        try:
            open_door = servo_queue.get()
            send_control_device(open_door)
            servo_queue.task_done()
            servo_open = open_door
        except Exception as e:
            print(f"Lỗi trong process_servo_queue: {e}")

def process_attendance_queue():
    """Xử lý hàng đợi gửi yêu cầu điểm danh."""
    while True:
        try:
            name, img, current_time = attendance_queue.get()
            handle_request_with_delay(name, img, current_time)
            attendance_queue.task_done()
        except Exception as e:
            print(f"Lỗi trong process_attendance_queue: {e}")

threading.Thread(target=process_servo_queue, daemon=True).start()
threading.Thread(target=process_attendance_queue, daemon=True).start()    

async def video_stream_from_esp32():
    global socket_running, servo_open
    uri = "ws://192.168.39.137:81"
    try:
        print(f"Connecting to ESP32 CAM at {uri}")
        async with websockets.connect(uri) as websocket:
            print("Connected to ESP32 CAM.")
            recognition_counter = 0
            last_recognized_frame = None
            frame_counter = 0
            while socket_running:
                frame_data = await websocket.recv()
                img_bytes = frame_data
                img_np = np.frombuffer(img_bytes, dtype=np.uint8)
                img = cv2.imdecode(img_np, cv2.IMREAD_COLOR)
                
                if img is None:
                    print("Corrupt JPEG data: premature end of data segment")
                    continue
                
                imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
                imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)
                facesCurFrame = face_recognition.face_locations(imgS)

                if recognition_counter % 6 == 0:
                    encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)
                    face_detected = False
                    name = "Unknown"
                    minFaceDis = 0
                    accuracy = 0

                    with camera_lock:
                        for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
                            matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
                            faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
                            matchIndex = np.argmin(faceDis)
                            minFaceDis = min(faceDis)

                            if matches[matchIndex] and (1 - minFaceDis) >= 0.60:
                                name = classNames[matchIndex].upper()
                                face_detected = True
                                accuracy = (1 - minFaceDis) * 100

                            y1, x2, y2, x1 = faceLoc
                            y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
                            height_increase = 20
                            y2 += height_increase
                            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            cv2.rectangle(img, (x1, y2 - 35), (x2, y2 + 20), (0, 255, 0), cv2.FILLED)
                            cv2.putText(img, f'id = {name}', (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 0.7, (255, 255, 255), 2)
                            cv2.putText(img, f'Accuracy: {accuracy:.2f}%', (x1 + 6, y2 + 10), cv2.FONT_HERSHEY_COMPLEX, 0.7, (255, 255, 255), 2)

                        if face_detected and name != "Unknown":
                            if not servo_open:
                                servo_queue.put(True)  
                            frame_counter += 1
                            if frame_counter >= 4:
                                attendance_queue.put((name, img, datetime.datetime.now()))  
                                frame_counter = 0
                        else:
                            if servo_open:
                                servo_queue.put(False)  

                        last_recognized_frame = img.copy()
                else:
                    if last_recognized_frame is not None:
                        img = last_recognized_frame.copy()
                    else:
                        for faceLoc in facesCurFrame:
                            y1, x2, y2, x1 = faceLoc
                            y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
                            height_increase = 20
                            y2 += height_increase
                            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

                ret, jpeg = cv2.imencode('.jpg', img)
                frame_bytes = jpeg.tobytes()

                try:
                    frame_queue.put(frame_bytes, block=False)
                except queue.Full:
                    pass

                recognition_counter += 1

    except websockets.exceptions.ConnectionClosedError:
        print("Connection to ESP32 CAM closed.")
    except Exception as e:
        print(f"Error in video stream: {e}")
    finally:
        print("Video stream ended.")
        socket_running = False

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

@app.route('/video_feed')
def video_feed():
    global socket_running, socket_thread
    if not socket_running:
        socket_running = True
        socket_thread = threading.Thread(target=asyncio.run, args=(video_stream_from_esp32(),))
        socket_thread.start()
    return Response(generate_video_feed(), mimetype='multipart/x-mixed-replace; boundary=frame')

def update_encodings_for_new_images(images_base64, className):
    global encodeListKnown, classNames
    className = str(className)

    images = [cv2.imdecode(np.frombuffer(base64.b64decode(img_str), np.uint8), cv2.IMREAD_COLOR) for img_str in images_base64]
    new_encodeListKnown = findEncodings(images)
    encodeListKnown.extend(new_encodeListKnown)
    classNames.extend([className] * len(new_encodeListKnown))
    print(f"Đã đăng ký thành công cho Id: {className}")

@app.route('/reload_encodings', methods=['POST'])
def reload_encodings():
    global encodeListKnown, classNames
    try:
        print("Đang tải lại dữ liệu nhận diện...")
        data = request.get_json()
        images_base64 = data['images']
        className = data.get('className')
        update_encodings_for_new_images(images_base64, className)
        return jsonify({"message": "Dữ liệu nhận diện đã được cập nhật."}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/close_connection', methods=['POST', 'GET'])
def close_connection():
    global socket_running
    if socket_running:
        socket_running = False
        return jsonify({"message": "Connection closed successfully."}), 200
    else:
        return jsonify({"message": "No active connection to close."}), 200

def run_flask():
    app.run(host='localhost', port=8888)

if __name__ == "__main__":
    flask_thread = threading.Thread(target=run_flask)
    flask_thread.start()
