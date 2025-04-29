import os
import cv2
import json
import sqlite3
from datetime import datetime

import uvicorn
from fastapi import FastAPI, UploadFile, File, Request
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

    conn.commit()
    conn.close()


init_db()


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/upload")
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


@app.get("/export/excel")
async def export_excel():
    conn = sqlite3.connect('truck_counter.db')
    df = pd.read_sql_query("SELECT * FROM detections", conn)
    conn.close()

    os.makedirs("static/reports", exist_ok=True)

    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    excel_path = f"static/reports/truck_report_{current_time}.xlsx"
    df.to_excel(excel_path, index=False)

    return FileResponse(excel_path, filename=f"truck_report_{current_time}.xlsx")


@app.get("/export/pdf")
async def export_pdf():
    conn = sqlite3.connect('truck_counter.db')
    c = conn.cursor()
    c.execute("SELECT timestamp, filename, truck_count FROM detections")
    rows = c.fetchall()
    conn.close()

    # Создаем директорию для отчетов, если её нет
    os.makedirs("static/reports", exist_ok=True)

    # Добавляем дату и время к имени файла
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    pdf_path = f"static/reports/truck_report_{current_time}.pdf"
    c = canvas.Canvas(pdf_path, pagesize=letter)

    c.setFont("Helvetica-Bold", 16)
    c.drawString(100, 750, "Truck Detection Report")
    c.setFont("Helvetica", 12)
    c.drawString(100, 730, f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    y = 700
    x_timestamp = 100
    x_filename = 300
    x_truck_count = 450
    c.drawString(x_timestamp, y, "Timestamp")
    c.drawString(x_filename, y, "Filename")
    c.drawString(x_truck_count, y, "Truck Count")

    y -= 20
    for row in rows:
        dt = datetime.fromisoformat(row[0])
        formatted_date = dt.strftime("%d.%m.%Y, %H:%M:%S")
        c.drawString(x_timestamp, y, formatted_date)
        c.drawString(x_filename, y, row[1])
        c.drawString(x_truck_count, y, str(row[2]))
        y -= 15
        if y < 50:
            c.showPage()
            y = 750

    c.save()

    return FileResponse(pdf_path, filename=f"truck_report_{current_time}.pdf")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)