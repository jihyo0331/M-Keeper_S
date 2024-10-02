# main.py
from fastapi import FastAPI, Request, UploadFile, File, Form, WebSocket, WebSocketDisconnect
from fastapi.templating import Jinja2Templates
from fastapi.responses import JSONResponse
import numpy as np
import base64
import io
from PIL import Image
from ultralytics import YOLO
import requests
import logging
import speech_recognition as sr

app = FastAPI()
templates = Jinja2Templates(directory="templates")

# YOLOv8n 모델 로드
model = YOLO('last.pt')

# 네이버 지도 API 설정
client_id = ''
client_secret = ''

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

clients = []

@app.get("/")
async def root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/navigate")
async def navigate_form(request: Request):
    return templates.TemplateResponse("navigate.html", {"request": request})


@app.get("/user")
async def user(request: Request):
    return templates.TemplateResponse("user.html", {"request": request})

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    clients.append(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            logger.info(f"Received message: {data}")
            if data == "send_g":
                for client in clients:
                    await client.send_text("G")
                    logger.info(f"Sent 'G' to {client}")
            elif data == "send_s":
                for client in clients:
                    await client.send_text("S")
                    logger.info(f"Sent 'S' to {client}")
    except WebSocketDisconnect:
        clients.remove(websocket)
        logger.info(f"Client disconnected: {websocket}")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        clients.remove(websocket)
    finally:
        await websocket.close()

@app.post("/swipe")
async def handle_swipe(request: Request):
    global swipe_detected
    data = await request.json()
    if data.get('swipe') == 'up':
        swipe_detected = True
        logger.info("Swipe up detected: swipe_detected is now True")
        return JSONResponse(content={"message": "Swipe up detected", "swipe_detected": swipe_detected})
    elif data.get('swipe') == 'left':
        swipe_detected = False
        logger.info("Swipe left detected: swipe_detected is now False")
        return JSONResponse(content={"message": "Swipe left detected", "swipe_detected": swipe_detected})
    return JSONResponse(content={"message": "No swipe detected"})

@app.get("/status")
async def get_status():
    return JSONResponse(content={"swipe_detected": swipe_detected})

@app.post("/navigate")
async def navigate(request: Request, origin_lat: float = Form(...), origin_lng: float = Form(...)):
    destination = get_destination_from_voice()
    if destination is None:
        return JSONResponse(content={"message": "Failed to recognize the destination location"}, status_code=400)

    logger.info(f"Origin coordinates: {origin_lat}, {origin_lng}")
    logger.info(f"Destination: {destination}")

    # 목적지의 좌표 변환
    def get_coordinates(address):
        url = f"https://naveropenapi.apigw.ntruss.com/map-geocode/v2/geocode"
        headers = {
            'X-NCP-APIGW-API-KEY-ID': client_id,
            'X-NCP-APIGW-API-KEY': client_secret
        }
        params = {'query': address}
        response = requests.get(url, headers=headers, params=params)
        response_json = response.json()
        logger.info(f"Geocoding response: {response_json}")  # 응답 결과 로깅
        if response_json['status'] == 'OK' and response_json['addresses']:
            location = response_json['addresses'][0]
            return location['y'], location['x']
        return None, None

    destination_lat, destination_lng = get_coordinates(destination)

    if not destination_lat:
        return JSONResponse(content={"message": "Failed to geocode the destination"}, status_code=400)

    logger.info(f"Destination coordinates: {destination_lat}, {destination_lng}")

    # 경로 탐색 (자동차 경로)
    directions_url = "https://naveropenapi.apigw.ntruss.com/map-direction/v1/driving"
    params = {
        'start': f"{origin_lng},{origin_lat}",
        'goal': f"{destination_lng},{destination_lat}",
        'option': 'trafast'

    }
    headers = {
        'X-NCP-APIGW-API-KEY-ID': client_id,
        'X-NCP-APIGW-API-KEY': client_secret
    }
    response = requests.get(directions_url, headers=headers, params=params)
    directions_result = response.json()

    logger.info(f"Directions result: {directions_result}")

    # 경로 정보 구성
    if directions_result['code'] != 0:
        return JSONResponse(content={"message": "No route found"}, status_code=400)

    try:
        route = directions_result['route']['traoptimal'][0]
        summary = route.get('summary', {})
        path = route.get('path', [])
        sections = route.get('section', [])
        guides = route.get('guide', [])
    except (IndexError, KeyError) as e:
        logger.error(f"Error processing directions result: {e}")
        return JSONResponse(content={"message": "Failed to process directions"}, status_code=500)

    directions = {
        "summary": summary,
        "path": path,
        "sections": sections,
        "guides": guides
    }

    return JSONResponse(content={"directions": directions}, status_code=200)

@app.post("/detect_objects")
async def detect_objects(request: Request):
    request_data = await request.json()
    image_data = request_data['image']
    image_data = image_data.split(",")[1]
    image = Image.open(io.BytesIO(base64.b64decode(image_data)))
    image = np.array(image)

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

    return JSONResponse(content=detection_result, status_code=200)

def get_destination_from_voice():
    recognizer = sr.Recognizer()
    microphone = sr.Microphone()

    with microphone as source:
        print("Please say the destination location:")
        recognizer.adjust_for_ambient_noise(source)
        audio = recognizer.listen(source)

    try:
        destination = recognizer.recognize_google(audio, language='ko-KR')
        print(f"You said: {destination}")
        return destination
    except sr.UnknownValueError:
        print("Google Speech Recognition could not understand audio")
        return None
    except sr.RequestError as e:
        print(f"Could not request results from Google Speech Recognition service; {e}")
        return None
