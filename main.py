import os
from threading import Lock

import cv2
import json
import sqlite3
from datetime import datetime

import uvicorn
from fastapi import FastAPI, UploadFile, File, Request, HTTPException
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.responses import StreamingResponse
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

    c.execute('''CREATE TABLE IF NOT EXISTS video_detections
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  timestamp TEXT,
                  video_name TEXT,
                  truck_id INTEGER,
                  appearance_time TEXT,
                  disappearance_time TEXT)''')

    c.execute('''CREATE TABLE IF NOT EXISTS stream_detections
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  stream_start TEXT,
                  stream_end TEXT,
                  truck_avg_count REAL,
                  frames_processed INTEGER)''')
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

    trucks_info = {}
    frame_count = 0
    MAX_MISSING_FRAMES = 10  # Максимальное количество кадров, которые грузовик может пропустить

    conn = sqlite3.connect('truck_counter.db')
    c = conn.cursor()

    request_time = datetime.now().isoformat()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        current_time = frame_count / fps
        frame_count += 1

        results = model.track(frame, persist=True, classes=[7])  # Только грузовики

        current_truck_ids = set()
        if results[0].boxes.id is not None:
            truck_ids = results[0].boxes.id.cpu().numpy().astype(int)
            boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
            confidences = results[0].boxes.conf.cpu().numpy()

            for truck_id, box, confidence in zip(truck_ids, boxes, confidences):
                current_truck_ids.add(truck_id)
                x1, y1, x2, y2 = box

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"Truck {truck_id} {confidence:.2f}",
                            (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                if truck_id not in trucks_info:
                    trucks_info[truck_id] = {
                        'first_seen': current_time,
                        'last_seen': current_time,
                        'missing_frames': 0
                    }
                    c.execute('''INSERT INTO video_detections
                                (timestamp, video_name, truck_id, appearance_time, disappearance_time)
                                VALUES (?, ?, ?, ?, ?)''',
                              (request_time, file.filename, int(truck_id),
                               current_time, None))
                else:
                    trucks_info[truck_id]['last_seen'] = current_time
                    trucks_info[truck_id]['missing_frames'] = 0  # Сбрасываем счетчик пропущенных кадров

        # Обрабатываем грузовики, которые не были обнаружены в текущем кадре
        for truck_id in list(trucks_info.keys()):
            if truck_id not in current_truck_ids:
                trucks_info[truck_id]['missing_frames'] += 1

                # Если грузовик не появлялся слишком долго, считаем его исчезнувшим
                if trucks_info[truck_id]['missing_frames'] > MAX_MISSING_FRAMES:
                    c.execute('''UPDATE video_detections
                                SET disappearance_time = ?
                                WHERE video_name = ? AND truck_id = ? AND timestamp = ? AND disappearance_time IS NULL''',
                              (trucks_info[truck_id]['last_seen'], file.filename, int(truck_id), request_time))
                    del trucks_info[truck_id]  # Удаляем из отслеживания

        out.write(frame)
        conn.commit()

    # После окончания видео фиксируем исчезновение оставшихся грузовиков
    for truck_id, info in trucks_info.items():
        c.execute('''UPDATE video_detections
                    SET disappearance_time = ?
                    WHERE video_name = ? AND truck_id = ? AND timestamp = ? AND disappearance_time IS NULL''',
                  (info['last_seen'], file.filename, int(truck_id), request_time))
    conn.commit()

    detection_data = []
    c.execute('''SELECT truck_id, appearance_time, disappearance_time 
                    FROM video_detections WHERE video_name = ? AND timestamp = ?''', (file.filename, request_time))
    for row in c.fetchall():
        detection_data.append({
            "truck_id": row[0],
            "appearance_time": row[1],
            "disappearance_time": row[2]
        })

    conn.close()
    cap.release()
    out.release()

    return JSONResponse({
        "message": "Video processed successfully",
        "processed_video": f"/static/results/processed_{file.filename}",
        "detection_data": detection_data
    })


video_lock = Lock()
# Глобальные переменные для управления стримами
active_streams = {}
stream_stats = {}


@app.get("/stream_feed")
async def stream_feed(request: Request, stream_id: str):
    if stream_id in active_streams:
        raise HTTPException(status_code=400, detail="Stream with this ID already exists")

    active_streams[stream_id] = True
    stream_stats[stream_id] = {
        'total_trucks': 0,
        'frame_count': 0,
        'start_time': datetime.now().isoformat()
    }

    def generate_frames():
        cap = cv2.VideoCapture(0)
        try:
            while active_streams.get(stream_id, False):
                ret, frame = cap.read()
                if not ret:
                    break

                results = model.track(frame, classes=[7])  # Только грузовики

                truck_count = 0
                if results[0].boxes.id is not None:
                    truck_count = len(results[0].boxes)

                with video_lock:
                    stream_stats[stream_id]['total_trucks'] += truck_count
                    stream_stats[stream_id]['frame_count'] += 1

                if results[0].boxes.id is not None:
                    boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
                    confidences = results[0].boxes.conf.cpu().numpy()

                    for box, confidence in zip(boxes, confidences):
                        x1, y1, x2, y2 = box
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(frame, f"Truck {confidence:.2f}",
                                    (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                ret, buffer = cv2.imencode('.jpg', frame)
                frame = buffer.tobytes()

                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        finally:
            cap.release()

    return StreamingResponse(generate_frames(), media_type="multipart/x-mixed-replace;boundary=frame")


@app.get("/stop_stream/{stream_id}")
async def stop_stream(stream_id: str):
    if stream_id in active_streams:
        active_streams[stream_id] = False
        stats = stream_stats[stream_id]
        if stats['frame_count'] > 0:
            avg_trucks = stats['total_trucks'] / stats['frame_count']
            end_time = datetime.now().isoformat()

            conn = sqlite3.connect('truck_counter.db')
            c = conn.cursor()
            c.execute('''INSERT INTO stream_detections
                          (stream_start, stream_end, truck_avg_count, frames_processed)
                          VALUES (?, ?, ?, ?)''',
                      (stats['start_time'], end_time, avg_trucks, stats['frame_count']))
            conn.commit()
            conn.close()

        del stream_stats[stream_id]

        return {"message": f"Stream {stream_id} stopped"}
    else:
        raise HTTPException(status_code=404, detail="Stream not found")


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


@app.get("/stream_history")
async def get_stream_history():
    conn = sqlite3.connect('truck_counter.db')
    c = conn.cursor()
    c.execute("SELECT * FROM stream_detections ORDER BY stream_start DESC")
    rows = c.fetchall()
    conn.close()

    history = []
    for row in rows:
        history.append({
            "id": row[0],
            "stream_start": row[1],
            "stream_end": row[2],
            "truck_avg_count": row[3],
            "frames_processed": row[4]
        })

    return JSONResponse(history)


@app.get("/export/excel/{table_name}")
async def export_excel(table_name: str):
    if not table_name.isidentifier():
        raise HTTPException(status_code=400, detail="Invalid table name")

    conn = sqlite3.connect('truck_counter.db')

    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    existing_tables = [table[0] for table in cursor.fetchall()]

    if table_name not in existing_tables:
        conn.close()
        raise HTTPException(status_code=404, detail="Table not found")

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

    os.makedirs("static/reports", exist_ok=True)

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


@app.get("/export/pdf/stream")
async def export_pdf_stream():
    conn = sqlite3.connect('truck_counter.db')
    c = conn.cursor()
    c.execute("SELECT id, stream_start, stream_end, truck_avg_count, frames_processed FROM stream_detections")
    rows = c.fetchall()
    conn.close()

    os.makedirs("static/reports", exist_ok=True)
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"stream_detections_report_{current_time}.pdf"
    pdf_path = f"static/reports/{filename}"

    c = canvas.Canvas(pdf_path, pagesize=letter)
    c.setFont("Helvetica-Bold", 16)
    c.drawString(100, 750, "Stream Detection Report")
    c.setFont("Helvetica", 12)
    c.drawString(100, 730, f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    y = 700
    x_id = 50
    x_start = x_id + 30
    x_end = x_start + 150
    x_avg = x_end + 130
    x_frames = x_avg + 80

    # Headers
    c.drawString(x_id, y, "ID")
    c.drawString(x_start, y, "Start Time")
    c.drawString(x_end, y, "End Time")
    c.drawString(x_avg, y, "Avg Trucks")
    c.drawString(x_frames, y, "Frames")

    y -= 20
    for row in rows:
        c.drawString(x_id, y, str(row[0]))

        start_dt = datetime.fromisoformat(row[1])
        c.drawString(x_start, y, start_dt.strftime("%d.%m.%Y %H:%M:%S"))

        end_dt = datetime.fromisoformat(row[2])
        c.drawString(x_end, y, end_dt.strftime("%d.%m.%Y %H:%M:%S"))

        c.drawString(x_avg, y, f"{row[3]:.2f}")
        c.drawString(x_frames, y, str(row[4]))

        y -= 15
        if y < 50:
            c.showPage()
            y = 750

    c.save()
    return FileResponse(pdf_path, filename=filename)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
