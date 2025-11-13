import cv2
from ultralytics import YOLO
import numpy as np
from collections import defaultdict


AVG_OBJECT_WIDTH_METERS = 1.7 # Средняя ширина авто
MAX_PIXEL_DISTANCE_PER_FRAME = 50 
MODEL_PATH = 'best-3.pt'
VIDEO_PATH = "inputs/cars1.mp4"

model = YOLO(MODEL_PATH)
cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    print(f"Ошибка: не удалось открыть видеофайл {VIDEO_PATH}")
    exit()

fps = cap.get(cv2.CAP_PROP_FPS)
dt = 1 / fps 

prev_positions_pixel = {}
speed_buffer = defaultdict(lambda: [])


while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    results = model.track(frame, persist=True, verbose=False)

    if results[0].boxes is not None and results[0].boxes.id is not None:
        boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
        track_ids = results[0].boxes.id.cpu().numpy().astype(int)
        class_ids = results[0].boxes.cls.cpu().numpy().astype(int)

        for box, track_id, class_id in zip(boxes, track_ids, class_ids):
            class_name = model.names[class_id]
            x1, y1, x2, y2 = box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"Class: {class_name}", (x1, y2 + 20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

            if class_name in ['car', 'truck', 'bus', 'motorcycle']:
                center_x_pixel = int((x1 + x2) / 2)
                center_y_pixel = int((y1 + y2) / 2)

                speed_kmh = 0
                if track_id in prev_positions_pixel:
                    prev_x_pixel, prev_y_pixel = prev_positions_pixel[track_id]
                    pixel_distance = np.sqrt((center_x_pixel - prev_x_pixel)**2 + (center_y_pixel - prev_y_pixel)**2)
                    if pixel_distance > MAX_PIXEL_DISTANCE_PER_FRAME:
                        prev_positions_pixel[track_id] = (center_x_pixel, center_y_pixel)
                        continue 
                    pixel_width = x2 - x1
                    
                    pixels_per_meter = pixel_width / AVG_OBJECT_WIDTH_METERS
                    prev_x_pixel, prev_y_pixel = prev_positions_pixel[track_id]

                    #pixel_distance = abs(center_x_pixel - prev_x_pixel)
                    #pixel_distance = np.sqrt((center_x_pixel - prev_x_pixel)**2 + (center_y_pixel - prev_y_pixel)**2)
                    pixel_distance = abs(center_x_pixel - prev_x_pixel)

                    meter_distance = pixel_distance / pixels_per_meter
                    speed_mps = meter_distance / dt
                    speed_kmh = speed_mps * 3.6
                    speed_buffer[track_id].append(speed_kmh)
                    smoothed_speed_kmh = np.mean(speed_buffer[track_id])

                    label = f"ID {track_id}: {smoothed_speed_kmh:.1f} km/h"
                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            prev_positions_pixel[track_id] = (center_x_pixel, center_y_pixel)

    cv2.imshow("Speed", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()