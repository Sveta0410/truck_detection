import os

import cv2
import json
import sqlite3
from datetime import datetime

import uvicorn
from fastapi import FastAPI, UploadFile, File, Request, HTTPException
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from ultralytics import YOLO
import pandas as pd
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

model = YOLO('yolo12n.pt')


def init_db():
    conn = sqlite3.connect('truck_counter.db')
    c = conn.cursor()

    c.execute('''CREATE TABLE IF NOT EXISTS detections
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  timestamp TEXT,
                  filename TEXT,
                  truck_count INTEGER,
                  detection_data TEXT)''')
    # Новая таблица для трекинга грузовиков в видео
    c.execute('''CREATE TABLE IF NOT EXISTS video_detections
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  timestamp TEXT,
                  video_name TEXT,
                  truck_id INTEGER,
                  appearance_time TEXT,
                  disappearance_time TEXT)''')
    conn.commit()
    conn.close()


init_db()


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/upload_image")
async def upload_image(file: UploadFile = File(...)):
    os.makedirs("static/uploads", exist_ok=True)
    os.makedirs("static/results", exist_ok=True)

    file_path = f"static/uploads/{file.filename}"
    with open(file_path, "wb") as buffer:
        buffer.write(await file.read())

    results = model(file_path)

    truck_count = 0
    detection_data = []

    for result in results:
        boxes = result.boxes
        for box in boxes:
            class_id = int(box.cls)
            if class_id == 7:  # в COCO класс 7 - грузовик
                truck_count += 1
                detection_data.append({
                    "class": "truck",
                    "confidence": float(box.conf),
                    "bbox": box.xyxy.tolist()[0]
                })

    img = cv2.imread(file_path)
    for detection in detection_data:
        bbox = detection['bbox']
        cv2.rectangle(img, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 255, 0), 2)
        cv2.putText(img, f"Truck {detection['confidence']:.2f}", (int(bbox[0]), int(bbox[1]) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    result_path = f"static/results/{file.filename}"
    cv2.imwrite(result_path, img)

    conn = sqlite3.connect('truck_counter.db')
    c = conn.cursor()
    c.execute("INSERT INTO detections (timestamp, filename, truck_count, detection_data) VALUES (?, ?, ?, ?)",
              (datetime.now().isoformat(), file.filename, truck_count, json.dumps(detection_data)))
    conn.commit()
    conn.close()

    return JSONResponse({
        "truck_count": truck_count,
        "original_image": f"/static/uploads/{file.filename}",
        "processed_image": f"/static/results/{file.filename}",
        "detections": detection_data
    })


@app.get("/history")
async def get_history():
    conn = sqlite3.connect('truck_counter.db')
    c = conn.cursor()
    c.execute("SELECT * FROM detections ORDER BY timestamp DESC")
    rows = c.fetchall()
    conn.close()

    history = []
    for row in rows:
        history.append({
            "id": row[0],
            "timestamp": row[1],
            "filename": row[2],
            "truck_count": row[3],
            "detection_data": json.loads(row[4])
        })

    return JSONResponse(history)


@app.get("/video_history")
async def get_history():
    conn = sqlite3.connect('truck_counter.db')
    c = conn.cursor()
    c.execute("SELECT * FROM video_detections ORDER BY timestamp DESC")
    rows = c.fetchall()
    conn.close()

    history = []
    for row in rows:
        history.append({
            "id": row[0],
            "timestamp": row[1],
            "video_name": row[2],
            "truck_id": row[3],
            "appearance_time": row[4],
            "disappearance_time": row[5]
        })

    return JSONResponse(history)


@app.get("/export/excel/{table_name}")
async def export_excel(table_name: str):
    if not table_name.isidentifier():
        raise HTTPException(status_code=400, detail="Invalid table name")

    conn = sqlite3.connect('truck_counter.db')
    # Получаем список существующих таблиц в базе данных
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    existing_tables = [table[0] for table in cursor.fetchall()]

    # Проверяем, существует ли запрашиваемая таблица
    if table_name not in existing_tables:
        conn.close()
        raise HTTPException(status_code=404, detail="Table not found")

    # Если таблица существует, читаем данные
    df = pd.read_sql_query(f"SELECT * FROM {table_name}", conn)
    conn.close()

    os.makedirs("static/reports", exist_ok=True)

    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"truck_{table_name}_report_{current_time}.xlsx"
    excel_path = f"static/reports/{filename}"
    df.to_excel(excel_path, index=False)

    return FileResponse(excel_path, filename=f"{filename}")


@app.get("/export/pdf/image")
async def export_pdf_image():
    conn = sqlite3.connect('truck_counter.db')
    c = conn.cursor()
    c.execute("SELECT id, timestamp, filename, truck_count FROM detections")
    rows = c.fetchall()
    conn.close()

    # Создаем директорию для отчетов, если её нет
    os.makedirs("static/reports", exist_ok=True)

    # Добавляем дату и время к имени файла
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"truck_detections_report_{current_time}.pdf"
    pdf_path = f"static/reports/{filename}"
    c = canvas.Canvas(pdf_path, pagesize=letter)

    c.setFont("Helvetica-Bold", 16)
    c.drawString(100, 750, "Truck Detection Report")
    c.setFont("Helvetica", 12)
    c.drawString(100, 730, f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    y = 700
    x_id = 100
    x_timestamp = 150
    x_filename = 350
    x_truck_count = 500
    c.drawString(x_id, y, "id")
    c.drawString(x_timestamp, y, "Timestamp")
    c.drawString(x_filename, y, "Filename")
    c.drawString(x_truck_count, y, "Truck Count")

    y -= 20
    for row in rows:
        c.drawString(x_id, y, str(row[0]))
        dt = datetime.fromisoformat(row[1])
        formatted_date = dt.strftime("%d.%m.%Y, %H:%M:%S")
        c.drawString(x_timestamp, y, formatted_date)
        c.drawString(x_filename, y, row[2])
        c.drawString(x_truck_count, y, str(row[3]))
        y -= 15
        if y < 50:
            c.showPage()
            y = 750

    c.save()

    return FileResponse(pdf_path, filename=f"{filename}")


@app.get("/export/pdf/video")
async def export_pdf_video():
    conn = sqlite3.connect('truck_counter.db')
    c = conn.cursor()
    c.execute("SELECT id, timestamp, video_name, truck_id, appearance_time, disappearance_time FROM video_detections")
    rows = c.fetchall()
    conn.close()

    # Создаем директорию для отчетов, если её нет
    os.makedirs("static/reports", exist_ok=True)

    # Добавляем дату и время к имени файла
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"truck_video_detections_report_{current_time}.pdf"
    pdf_path = f"static/reports/{filename}"
    c = canvas.Canvas(pdf_path, pagesize=letter)

    c.setFont("Helvetica-Bold", 16)
    c.drawString(100, 750, "Truck Video Detection Report")
    c.setFont("Helvetica", 12)
    c.drawString(100, 730, f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    y = 700
    x_id = 10
    x_timestamp = x_id + 15
    x_video_name = x_timestamp + 120
    x_truck_id = x_video_name + 185
    x_appearance_time = x_truck_id + 50
    x_disappearance_time = x_appearance_time + 110
    c.drawString(x_id, y, "id")
    c.drawString(x_timestamp, y, "Timestamp")
    c.drawString(x_video_name, y, "Video name")
    c.drawString(x_truck_id, y, "Truck id")
    c.drawString(x_appearance_time, y, "Appearance time")
    c.drawString(x_disappearance_time, y, "Disappearance time")

    y -= 20
    for row in rows:
        c.drawString(x_id, y, str(row[0]))
        dt = datetime.fromisoformat(row[1])
        formatted_date = dt.strftime("%d.%m.%Y, %H:%M:%S")
        c.drawString(x_timestamp, y, formatted_date)
        c.drawString(x_video_name, y, row[2])
        c.drawString(x_truck_id, y, str(row[3]))
        c.drawString(x_appearance_time, y, row[4])
        c.drawString(x_disappearance_time, y, row[5])
        y -= 15
        if y < 50:
            c.showPage()
            y = 750

    c.save()

    return FileResponse(pdf_path, filename=f"{filename}")


@app.post("/process_video")
async def process_video(file: UploadFile = File(...)):
    os.makedirs("static/uploads", exist_ok=True)
    os.makedirs("static/results", exist_ok=True)

    input_path = f"static/uploads/{file.filename}"
    with open(input_path, "wb") as buffer:
        buffer.write(await file.read())

    output_path = f"static/results/processed_{file.filename}"

    cap = cv2.VideoCapture(input_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*'X264')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    truck_tracker = {}
    next_truck_id = 1
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        current_time = frame_count / fps
        frame_count += 1

        # обрабатываем каждый второй кадр (для ускорения процесса обработки)
        # if frame_count % 2 != 0:
        #     continue

        results = model(frame)

        current_truck_positions = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                if int(box.cls) == 7:  # Только грузовики
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    x_center = (x1 + x2) // 2
                    y_center = (y1 + y2) // 2

                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                    matched = False
                    for truck_id, data in truck_tracker.items():
                        last_pos = data['last_position']
                        dist = ((x_center - last_pos[0]) ** 2 + (y_center - last_pos[1]) ** 2) ** 0.5
                        if dist < 50:  # Пороговое расстояние для сопоставления
                            truck_tracker[truck_id]['last_position'] = (x_center, y_center)
                            truck_tracker[truck_id]['last_seen'] = current_time
                            current_truck_positions.append((x_center, y_center))
                            cv2.putText(frame, f"Truck {truck_id}", (x1, y1 - 10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                            matched = True
                            break

                    if not matched:
                        truck_id = next_truck_id
                        next_truck_id += 1
                        truck_tracker[truck_id] = {
                            'first_seen': current_time,
                            'last_seen': current_time,
                            'last_position': (x_center, y_center)
                        }
                        cv2.putText(frame, f"Truck {truck_id}", (x1, y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        out.write(frame)

    cap.release()
    out.release()

    detection_data = []

    conn = sqlite3.connect('truck_counter.db')
    c = conn.cursor()
    for truck_id, data in truck_tracker.items():
        detection_data.append({
            "truck_id": truck_id,
            "appearance_time": data['first_seen'],
            "disappearance_time": data['last_seen']
        })

        c.execute('''INSERT INTO video_detections
                     (timestamp, video_name, truck_id, appearance_time, disappearance_time)
                     VALUES (?, ?, ?, ?, ?)''',
                  (datetime.now().isoformat(), file.filename, truck_id,
                   data['first_seen'], data['last_seen']))
    conn.commit()
    conn.close()

    return JSONResponse({
        "message": "Video processed successfully",
        "processed_video": f"/static/results/processed_{file.filename}",
        # "trucks_detected": len(truck_tracker),
        "detection_data": detection_data
    })


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
