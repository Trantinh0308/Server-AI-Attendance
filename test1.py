import threading
import time
import cv2
import numpy as np
import face_recognition
import os
from flask import Flask, Response, jsonify, request
import pickle
import datetime
import base64
import requests
from flask_cors import CORS
app = Flask(__name__)
CORS(app)

TRAINING_IMAGES_PATH = 'Training_images'
ENCODINGS_PATH = 'encodings.pickle' 

camera_lock = threading.Lock()

def load_training_images():
    """
    Tải và trả về danh sách ảnh huấn luyện và các tên của người (lớp).
    """
    images = []  # Danh sách lưu ảnh
    classNames = []  # Danh sách lưu tên các lớp (mỗi thư mục con là một lớp)
    myList = os.listdir(TRAINING_IMAGES_PATH)  # Lấy danh sách các thư mục con

    # Lọc để chỉ lấy các thư mục con, không lấy các tệp tin
    myList = [f for f in myList if os.path.isdir(os.path.join(TRAINING_IMAGES_PATH, f))]
    
    for person_name in myList:
        person_path = os.path.join(TRAINING_IMAGES_PATH, person_name)  # Đường dẫn đến thư mục của mỗi người
        person_images = os.listdir(person_path)  # Lấy danh sách ảnh trong thư mục của người đó
        
        for image_name in person_images:
            curImg = cv2.imread(os.path.join(person_path, image_name))  # Đọc ảnh
            if curImg is not None:
                images.append(curImg)  # Thêm ảnh vào danh sách images
                classNames.append(person_name)  # Thêm tên người vào classNames

    return images, classNames

def load_encodings():
    """
    Tải các encoding từ tệp tin (nếu có).
    """
    if os.path.exists(ENCODINGS_PATH):
        with open(ENCODINGS_PATH, 'rb') as f:
            encodeDict = pickle.load(f)
        return encodeDict
    else:
        return None


def find_encodings(images, classNames):  # dky
    """
    Nhận diện và trả về encoding của các khuôn mặt trong các ảnh.
    """
    encodeDict = {}  # Từ điển lưu trữ encoding với tên lớp
    for img, person_name in zip(images, classNames):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Chuyển đổi ảnh sang RGB
        encode = face_recognition.face_encodings(img)  # Lấy encoding của khuôn mặt
        if len(encode) > 0:  # Kiểm tra xem có phát hiện khuôn mặt không
            encodeDict[person_name] = encode[0]  # Lưu encoding của khuôn mặt cùng với tên người
    return encodeDict

def save_encodings(encodeDict):  # dky
    """
    Lưu các encoding đã nhận diện vào tệp tin.
    """
    with open(ENCODINGS_PATH, 'wb') as f:
        pickle.dump(encodeDict, f)
    print("Encodings have been saved.")

data_sending_thread = None

# Hàm gửi yêu cầu attendance request
def send_attendance_request(name, img, current_time):
    try:
        # Mã hóa ảnh thành base64
        _, img_encoded = cv2.imencode('.jpg', img)
        img_base64 = base64.b64encode(img_encoded).decode('utf-8')

        # Tạo payload
        payload = {
            "employeeId": name,  # Tên của nhân viên
            "imageCode": img_base64,
            "date": current_time.strftime("%Y-%m-%d"),
            "time": current_time.strftime("%H:%M:%S")
        }

        api_url = 'http://localhost:8686/api/attendance'
        # Gửi POST request
        response = requests.post(api_url, json=payload)

        # Kiểm tra mã trạng thái
        if response.status_code == 200 or response.status_code == 201:
            print("Attendance recorded successfully!")
        else:
            print(f"Failed to record attendance. Server returned: {response.status_code}")
            print("Response details:", response.text)

    except requests.exceptions.RequestException as e:
        print("Error sending request:", e)

# Hàm thực hiện delay trước khi gửi request
last_name = ""
last_request_time = 0

def handle_request_with_delay(name, img, current_time):
    global last_name, last_request_time
    current_time_in_seconds = time.time()  # Thời gian hiện tại tính bằng giây

    # Kiểm tra nếu name khác last_name hoặc đã đủ 10 phút kể từ request cuối cùng
    if name != last_name or (current_time_in_seconds - last_request_time) >= 600:  
        print(f"Sending attendance request for {name} at {current_time.strftime('%H:%M:%S')}")
        send_attendance_request(name, img, current_time)
        
        # Cập nhật last_name và last_request_time sau khi gửi request
        last_name = name
        last_request_time = current_time_in_seconds

def showCamera():
    global data_sending_thread
    encodeListKnown = load_encodings()
    images, classNames = load_training_images()

    if encodeListKnown is None:
        encodeListKnown = find_encodings(images, classNames)
        save_encodings(encodeListKnown)
        print('Encoding Complete') 

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    while True:
        ret, img = cap.read()
        if not ret:
            print("Error: Failed to capture image.")
            break  

        imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)  
        imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

        current_time = datetime.datetime.now()
        time_str = current_time.strftime("%d-%m-%Y %H:%M:%S")

        cv2.putText(img, f'{time_str}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)

        facesCurFrame = face_recognition.face_locations(imgS)
        encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)

        face_detected = False
        name = "Unknown"
        minFaceDis = 0

        for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
            matches = face_recognition.compare_faces(list(encodeListKnown.values()), encodeFace)
            faceDis = face_recognition.face_distance(list(encodeListKnown.values()), encodeFace)

            matchIndex = np.argmin(faceDis)
            minFaceDis = min(faceDis)

            if matches[matchIndex] and (1 - minFaceDis) >= 0.65:
                name = classNames[matchIndex].upper()
                face_detected = True

            y1, x2, y2, x1 = faceLoc
            y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
            height_increase = 20
            y2 += height_increase
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.rectangle(img, (x1, y2 - 35), (x2, y2 + 20), (0, 255, 0), cv2.FILLED)
            cv2.putText(img, f'id = {name}', (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 0.7, (255, 255, 255), 2)

        if face_detected and name != "Unknown":
            if data_sending_thread is None or not data_sending_thread.is_alive():  
                data_sending_thread = threading.Thread(target=handle_request_with_delay, args=(name, img, current_time))
                data_sending_thread.start()

        ret, jpeg = cv2.imencode('.jpg', img)
        frame = jpeg.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()
    
@app.route('/camera')
def video_feed():
    return Response(showCamera(), mimetype='multipart/x-mixed-replace; boundary=frame')


def save_images(images, className):
    person_path = os.path.join(TRAINING_IMAGES_PATH, str(className))
    if not os.path.exists(person_path):
        os.makedirs(person_path)

    for i, img in enumerate(images):
        image_name = f'{className}_{i+1}.jpg'
        cv2.imwrite(os.path.join(person_path, image_name), img)

def register_new_face(images, className):
    """
    Hàm đăng ký một nhóm ảnh mới từ client với một classname.
    Tham số truyền vào gồm danh sách ảnh và tên lớp.
    """
    encodeDict = load_encodings() if load_encodings() else {}  
    new_encodings = find_encodings(images, [className] * len(images))  

    encodeDict.update(new_encodings)

    save_encodings(encodeDict)
    print(f"Face for {className} has been registered successfully.")        


capture_done = False  # Cờ để kiểm tra xem quá trình chụp ảnh đã hoàn thành chưa
cap = None  # Biến để lưu đối tượng video capture
employeeId = -1
# Hàm chụp ảnh
def capture_images():
    global cap, capture_done,employeeId
    with camera_lock:  # Đảm bảo không có truy cập đồng thời vào camera
        start_time = time.time()  # Lưu thời gian bắt đầu
        captured_images = []  # Danh sách lưu các ảnh đã chụp
        while len(captured_images) < 10:
            ret, frame = cap.read()
            if not ret:
                print("Error: Failed to capture image.")
                break

            elapsed_time = time.time() - start_time
            if elapsed_time >= 2 and len(captured_images) < 10:
                # Lấy một ảnh và thêm vào danh sách
                captured_images.append(frame)
                print(f"Captured image {len(captured_images)}")
                time.sleep(1)  # Chờ 1 giây trước khi chụp ảnh tiếp theo

        capture_done = True  # Đánh dấu là đã hoàn thành quá trình chụp ảnh

        save_images(captured_images, employeeId)
        register_new_face(captured_images, employeeId)
        employeeId = -1
        print("Captured 10 images.")
        cap.release()  # Giải phóng camera sau khi chụp ảnh xong
        print("Camera has been released.")

# Hàm phát video stream liên tục
def generate_video_feed():
    global cap
    while not capture_done:  # Tiếp tục phát video cho đến khi quá trình chụp ảnh kết thúc
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture image.")
            break

        # Mã hóa ảnh thành JPEG
        ret, jpeg = cv2.imencode('.jpg', frame)
        if ret:
            frame_bytes = jpeg.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    # Sau khi hoàn thành việc chụp ảnh, dừng video stream và giải phóng camera
    print("Stopping video stream as capture is done.")
    cap.release()

@app.route('/capture_images', methods=['POST'])
def registerId():
    global employeeId
    try:
        # Lấy dữ liệu từ yêu cầu JSON
        data = request.get_json()

        # Kiểm tra xem ID có trong dữ liệu hay không
        if 'id' not in data:
            return jsonify({'error': 'No ID provided'}), 400  # Trả về lỗi 400 nếu không có ID

        # Lưu ID vào biến global (hoặc cơ sở dữ liệu nếu cần)
        employeeId = data['id']

        # Trả về phản hồi thành công với ID đã nhận
        return jsonify({'message': 'ID received successfully', 'employeeId': employeeId}), 200

    except Exception as e:
        # Xử lý lỗi bất ngờ (nếu có)
        return jsonify({'error': f'An error occurred: {str(e)}'}), 500

@app.route('/capture_images', methods=['GET'])
def capture_images_api():
    global cap, capture_done

    capture_done = False
    cap = cv2.VideoCapture(0)  # Khởi tạo lại camera

    if not cap.isOpened():
        print("Error: Could not open camera.")
        return "Error: Could not open camera."

    # Chạy quá trình chụp ảnh trong một luồng riêng biệt
    capture_thread = threading.Thread(target=capture_images)
    capture_thread.start()

    # Trả về video stream cho client
    return Response(generate_video_feed(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(host='localhost', port=6996)
