import os
import cv2
import face_recognition
import base64
import requests

ATTENDANCE_URL = 'http://localhost:8686/api/attendance'

def load_training_images(imagesPath):
    images = []  
    classNames = [] 
    myList = os.listdir(imagesPath)
    myList = [f for f in myList if os.path.isdir(os.path.join(imagesPath, f))]  

    for person_name in myList:
        person_path = os.path.join(imagesPath, person_name)
        person_images = os.listdir(person_path)
        for image_name in person_images:
            img_path = os.path.join(person_path, image_name)
            curImg = cv2.imread(img_path)
            if curImg is not None:
                images.append(curImg)  
                classNames.append(person_name)  

    return images, classNames

def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)
        if encode:
            encodeList.append(encode[0])
    return encodeList

def send_attendance_request(name, img, current_time):
    try:
        _, img_encoded = cv2.imencode('.jpg', img)
        img_base64 = base64.b64encode(img_encoded).decode('utf-8')

        payload = {
            "employeeId": name,
            "imageCode": img_base64,
            "date": current_time.strftime("%Y-%m-%d"),
            "time": current_time.strftime("%H:%M:%S")
        }
        response = requests.post(ATTENDANCE_URL, json=payload)

        if response.status_code == 200 or response.status_code == 201:
            print("Attendance recorded successfully!")
        else:
            print(f"Failed to record attendance. Server returned: {response.status_code}")
            print("Response details:", response.text)

    except requests.exceptions.RequestException as e:
        print("Error sending request:", e)


