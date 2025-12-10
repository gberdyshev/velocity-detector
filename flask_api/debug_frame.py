import cv2
from ultralytics import YOLO

VIDEO_PATH = "../inputs/cars1.mp4" 
MODEL_PATH = 'blurry-f.pt'

# Время, cек кадра
TARGET_TIME = 0

model = YOLO(f"./ml/{MODEL_PATH}")

cap = cv2.VideoCapture(VIDEO_PATH)
fps = cap.get(cv2.CAP_PROP_FPS)
target_frame = int(TARGET_TIME * fps)

cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
ret, frame = cap.read()

if ret:
    results = model.track(frame, persist=True)
    boxes = results[0].boxes.xyxy.cpu().numpy()
    
    print(f"--- Objects on frame {target_frame} ---")
    for box in boxes:
        x1, y1, x2, y2 = map(int, box)
        print(f"Box: x1={x1}, y1={y1}, x2={x2}, y2={y2}")
        
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        text = f"{x1},{y1},{x2},{y2}"
        cv2.putText(frame, text, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.imwrite("debug_frame.jpg", frame)
else:
    print("Не удалось прочитать кадр")

cap.release()