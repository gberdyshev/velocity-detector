import cv2
import numpy as np
import pandas as pd
import logging
from ultralytics import YOLO

logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def calculate_iou(box1, box2):
    """
    Считает IoU для двух прямоугольников в формате [x1, y1, x2, y2].
    """
    # box1 - от пользователя, box2 - от YOLO
    
    x_left = max(box1[0], box2[0])
    y_top = max(box1[1], box2[1])
    x_right = min(box1[2], box2[2])
    y_bottom = min(box1[3], box2[3])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

    union_area = area1 + area2 - intersection_area
    if union_area == 0:
        return 0.0
        
    return intersection_area / union_area



def process_video_task(video_path, pixel_size, selected_time, target_box_dict):
    try:
        print("Начинаем...")
        logging.info("Начинаем")
        model = YOLO("./ml/blurry.pt")
        
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        target_frame_idx = int(selected_time * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame_idx)
        
        ret, frame = cap.read()
        if not ret:
            raise ValueError("Не удалось прочитать кадр по указанному времени")

        results = model.track(frame, persist=True)
        
        best_iou = 0
        target_track_id = None
        
        user_box = [
        target_box_dict['x1'], 
        target_box_dict['y1'], 
        target_box_dict['x2'], 
        target_box_dict['y2']
        ]
        
        if results[0].boxes.id is None:
             raise ValueError("На выбранном кадре трекер не нашел объектов")

        boxes_xyxy = results[0].boxes.xyxy.cpu().numpy()
        track_ids = results[0].boxes.id.int().cpu().numpy()


        for box, trk_id in zip(boxes_xyxy, track_ids):
            iou = calculate_iou(user_box, box)
            if iou > best_iou:
                best_iou = iou
                target_track_id = trk_id
        
        if target_track_id is None or best_iou < 0.3:
            raise ValueError("Не удалось сопоставить объект (низкий IoU или объект не найден)")



        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        
        raw_data = [] # (time, x, y)
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            current_time = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            
            res = model.track(frame, persist=True, verbose=False)
            
            if res[0].boxes.id is not None:
                boxes = res[0].boxes.xywh.cpu().numpy()
                ids = res[0].boxes.id.int().cpu().numpy()
                
                # Ищем наш ID
                mask = ids == target_track_id
                if np.any(mask):
                    box = boxes[mask][0]
                    raw_data.append({
                        'time': current_time,
                        'x_px': box[0],
                        'y_px': box[1]
                    })
        
        cap.release()
        
        if not raw_data:
            raise ValueError("Объект потерян сразу и данных нет")
            
        df = pd.DataFrame(raw_data)
        

        window_size = max(3, int(fps / 4)) 
        df['x_smooth'] = df['x_px'].rolling(window=window_size, center=True).mean()
        df['y_smooth'] = df['y_px'].rolling(window=window_size, center=True).mean()
        
        # Удаляем NaN от сглаживания
        df = df.dropna()
        
        df['x_m'] = df['x_smooth'] * pixel_size
        df['y_m'] = df['y_smooth'] * pixel_size
        
        df['dt'] = df['time'].diff()
        df['dx'] = df['x_m'].diff()
        df['dy'] = df['y_m'].diff()
        
        df['v_x'] = df['dx'] / df['dt']
        df['v_y'] = df['dy'] / df['dt']
        
        df['v'] = np.sqrt(df['v_x']**2 + df['v_y']**2)
        
        df['dv_x'] = df['v_x'].diff()
        df['dv_y'] = df['v_y'].diff()
        
        df['a_x'] = df['dv_x'] / df['dt']
        df['a_y'] = df['dv_y'] / df['dt']
        df['a'] = np.sqrt(df['a_x']**2 + df['a_y']**2)
        
        df = df.fillna(0)
        
        result_data = {
            'time': df['time'].tolist(),
            'x': df['x_m'].tolist(), # в метрах
            'y': df['y_m'].tolist(),
            'v_x': df['v_x'].tolist(),
            'v_y': df['v_y'].tolist(),
            'v': df['v'].tolist(),
            'a_x': df['a_x'].tolist(),
            'a_y': df['a_y'].tolist(),
            'a': df['a'].tolist(),
        }

        print(result_data)
        return result_data
        
        if os.path.exists(video_path):
            os.remove(video_path)

    except Exception as e:
        logging.error(e)
        print(f"Error: {e}")
