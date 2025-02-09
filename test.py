import threading
import cv2
import os
import face_recognition
import time
import datetime
import numpy as np
from flask import Flask, Response, jsonify, request
from flask_cors import CORS
import websockets
import asyncio
import base64
from utils import findEncodings, load_training_images, send_attendance_request, send_control_device

app = Flask(__name__)
CORS(app)

TRAINING_IMAGES_PATH = 'Training_images'
camera_lock = threading.Lock()
capture_done = False
cap = None
last_name = ""
last_request_time = 0
data_sending_thread = None
employeeId = -1
encodeListKnown = []
classNames = []

def init_encoding():
    global encodeListKnown, classNames
    images, classNames = load_training_images(TRAINING_IMAGES_PATH)
    encodeListKnown = findEncodings(images)
    print("Tải encodings thành công.")

init_encoding()

def handle_request_with_delay(name, img, current_time):
    global last_name, last_request_time
    current_time_in_seconds = time.time()

    if name != last_name or (current_time_in_seconds - last_request_time) >= 60:
        print(f"Sending attendance request for {name} at {current_time.strftime('%H:%M:%S')}")
        send_attendance_request(name, img, current_time)
        
        last_name = name
        last_request_time = current_time_in_seconds

async def video_stream(websocket):
    cap = cv2.VideoCapture(0)  

    if not cap.isOpened():
        print("Error: Could not open camera.")
        await websocket.close()
        return
    
    frame_counter = 0
    while True:
        ret, img = cap.read()
        if not ret:
            print("Error: Failed to capture image.")
            break

        imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)  
        imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)
        current_time = datetime.datetime.now()

        facesCurFrame = face_recognition.face_locations(imgS)
        encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)

        face_detected = False
        name = "Unknown"
        minFaceDis = 0
        accuracy = 0

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

        if face_detected and name != "Unknown" :
            frame_counter += 1  
            if frame_counter >= 4:
                handle_request_with_delay(name, img, current_time)
                send_control_device(True)
                frame_counter = 0 
        else :  
            send_control_device(False)
        ret, jpeg = cv2.imencode('.jpg', img)
        frame = base64.b64encode(jpeg.tobytes()).decode('utf-8')

        if websocket.open:
                try:
                    await websocket.send(frame)
                except websockets.exceptions.ConnectionClosedError:
                    print("Connection camera closed.")
                    break
                except Exception as e:
                    print(f"Error sending frame: {e}")
                    break
    cap.release()

async def start_websocket_server():
    server = await websockets.serve(video_stream, "localhost", 8080)
    await server.wait_closed()

def start_server_thread():
    asyncio.run(start_websocket_server())

def save_images(images, className):
    person_path = os.path.join(TRAINING_IMAGES_PATH, str(className))
    if not os.path.exists(person_path):
        os.makedirs(person_path)

    for i, img in enumerate(images):
        image_name = f'{className}_{i+1}.jpg'
        cv2.imwrite(os.path.join(person_path, image_name), img)
    update_encodings_for_new_images(images, className)

def update_encodings_for_new_images(images, className):
    global encodeListKnown, classNames
    className = str(className)
    new_encodeListKnown = findEncodings(images)
    encodeListKnown.extend(new_encodeListKnown)  
    classNames.extend([className] * len(new_encodeListKnown))
    print(f"Đã đăng ký thành công cho Id: {employeeId}")    

def capture_images():
    global cap, capture_done, employeeId
    with camera_lock:
        start_time = time.time()
        captured_images = []
        while len(captured_images) < 10:
            ret, frame = cap.read()
            if not ret:
                print("Error: Failed to capture image.")
                break

            elapsed_time = time.time() - start_time
            if elapsed_time >= 2 and len(captured_images) < 10:
                captured_images.append(frame)
                print(f"Captured image {len(captured_images)}")
                time.sleep(1)

        save_images(captured_images, employeeId)
        capture_done = True
        employeeId = -1
        cap.release()

def generate_video_feed():
    global cap
    while not capture_done:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture image.")
            break

        ret, jpeg = cv2.imencode('.jpg', frame)
        if ret:
            frame_bytes = jpeg.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    print("Dữ liệu khuôn mặt đã được lấy xong.")
    cap.release()

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
    global capture_done,cap
    capture_done = False
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open camera.")
        return "Error: Could not open camera."

    capture_thread = threading.Thread(target=capture_images)
    capture_thread.start()

    return Response(generate_video_feed(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    threading.Thread(target=start_server_thread, daemon=True).start()
    app.run(host='localhost', port=8888)
