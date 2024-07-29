# stream.py
import cv2
from ultralytics import YOLO
import numpy as np

# YOLOv8 모델 로드
model = YOLO('last.pt')


def getCameraStream():
    camera = cv2.VideoCapture(0)

    while True:
        retVal, frame = camera.read()
        if not retVal:
            break

        # YOLOv8 모델을 사용하여 객체 탐지
        results = model(frame)

        # 결과를 그리기
        for result in results:
            boxes = result.boxes.data.tolist()  # 바운딩 정보
            for box in boxes:
                x1, y1, x2, y2, score, class_id = map(int, box[:6])
                label = f'{model.names[class_id]} {score:.2f}'
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        retVal, jpgImg = cv2.imencode('.jpg', frame)

        jpgBin = bytearray(jpgImg.tobytes())

        yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + jpgBin + b'\r\n')


def detect_objects_in_image(image):
    results = model(image)

    detection_result = []
    for result in results:
        boxes = result.boxes.data.tolist()  # 바운딩 박스 정보 추출
        for box in boxes:
            x1, y1, x2, y2, score, class_id = map(int, box[:6])
            detection_result.append({
                "label": model.names[class_id],
                "score": score,
                "box": [x1, y1, x2, y2]
            })

    return detection_result
