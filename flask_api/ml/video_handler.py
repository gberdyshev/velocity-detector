import cv2
import numpy as np
import pandas as pd
import logging
from ultralytics import YOLO
from collections import defaultdict

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

    if x_right < x_left or y_bottom < y_top: return 0.0
    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = area1 + area2 - intersection_area
    if union_area == 0: return 0.0
    return intersection_area / union_area

def process_video_task(video_path, pixel_size, selected_time, target_box_dict):
    cap = None
    try:
        print(f"--- Начинаем обработку: {video_path} ---", flush=True)
        logging.info("Начинаем")
        
        model = YOLO("./ml/blurry.pt")
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
             raise ValueError("Не удалось открыть файл видео")

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        target_frame_idx = int(selected_time * fps)
        user_box = [
            target_box_dict['x1'], 
            target_box_dict['y1'], 
            target_box_dict['x2'], 
            target_box_dict['y2']
        ]


        all_tracks_history = defaultdict(list)
        selected_track_id = None
        

        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        current_frame_idx = 0
        
        print(f"Целевой кадр: {target_frame_idx} из {total_frames}", flush=True)

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            current_time = current_frame_idx / fps
            results = model.track(frame, persist=True, verbose=False)
            
            if results[0].boxes.id is not None:
                ids = results[0].boxes.id.int().cpu().numpy()
                boxes_xyxy = results[0].boxes.xyxy.cpu().numpy() 
                boxes_wh = results[0].boxes.xywh.cpu().numpy()
                
                for i, trk_id in enumerate(ids):
                    all_tracks_history[trk_id].append({
                        'time': current_time,
                        'x_px': boxes_wh[i][0],
                        'y_px': boxes_wh[i][1]
                    })
                
                if current_frame_idx == target_frame_idx:
                    print(f"--- Проверка кадра {current_frame_idx} ---", flush=True)
                    best_iou = 0
                    best_id = None
                    
                    for i, trk_id in enumerate(ids):
                        iou = calculate_iou(user_box, boxes_xyxy[i])
                        print(f"ID {trk_id}: IoU {iou:.4f}", flush=True)
                        if iou > best_iou:
                            best_iou = iou
                            best_id = trk_id
                    
                    if best_id is not None and best_iou > 0.1:
                        selected_track_id = best_id
                        print(f"!!! ОБЪЕКТ НАЙДЕН: ID {selected_track_id} !!!", flush=True)
                    else:
                        print("!!! ОБЪЕКТ НЕ НАЙДЕН НА ЦЕЛЕВОМ КАДРЕ !!!", flush=True)

            current_frame_idx += 1
        
        cap.release()
        
        if selected_track_id is None:
            raise ValueError("Не удалось сопоставить объект (низкий IoU или объект не найден на указанной секунде)")
            
        if selected_track_id not in all_tracks_history:
            raise ValueError("ID выбран, но история пуста (внутренняя ошибка логики)")
            
        raw_data = all_tracks_history[selected_track_id]
        
        if not raw_data:
            raise ValueError("Объект найден, но данных нет")
            
        df = pd.DataFrame(raw_data)
        
        smooth_window = max(5, int(fps)) 
        
        df['x_smooth'] = df['x_px'].rolling(window=smooth_window, center=True).mean()
        df['y_smooth'] = df['y_px'].rolling(window=smooth_window, center=True).mean()
        
        df = df.dropna()
        
        if df.empty:
            raise ValueError("Слишком короткий трек для анализа")

        df['x_m'] = df['x_smooth'] * pixel_size
        df['y_m'] = df['y_smooth'] * pixel_size
        
        df['dt'] = df['time'].diff()
        df = df[df['dt'] > 0]
        
        df['v_x'] = (df['x_m'].diff() / df['dt']).fillna(0)
        df['v_y'] = (df['y_m'].diff() / df['dt']).fillna(0)
        df['v'] = np.sqrt(df['v_x']**2 + df['v_y']**2)
        

        df['v_smooth'] = df['v'].rolling(window=smooth_window, center=True).mean()
        df['v_x_smooth'] = df['v_x'].rolling(window=smooth_window, center=True).mean()
        df['v_y_smooth'] = df['v_y'].rolling(window=smooth_window, center=True).mean()
        
        df['a'] = df['v_smooth'].diff() / df['dt']
        df['a_x'] = df['v_x_smooth'].diff() / df['dt']
        df['a_y'] = df['v_y_smooth'].diff() / df['dt']        

        # Погрешности
        df['x_std_px'] = df['x_px'].rolling(window=smooth_window, center=True).std()
        df['y_std_px'] = df['y_px'].rolling(window=smooth_window, center=True).std()
        df['err_x'] = df['x_std_px'] * pixel_size
        df['err_y'] = df['y_std_px'] * pixel_size

        df['err_v'] = df['v'].rolling(window=smooth_window, center=True).std()

        # Заполняем NaN нулями
        df = df.fillna(0.0)


        
        def safe_list(series): return series.tolist()

        result_data = {
            'time': safe_list(df['time']),
            'x': safe_list(df['x_m']),
            'y': safe_list(df['y_m']),
            'v': safe_list(df['v_smooth']),    
            'v_x': safe_list(df['v_x_smooth']),
            'v_y': safe_list(df['v_y_smooth']),
            'a': safe_list(df['a']),
            'a_x': safe_list(df['a_x']),
            'a_y': safe_list(df['a_y']),
            'err_x': safe_list(df['err_x']),
            'err_y': safe_list(df['err_y']),
            'err_v': safe_list(df['err_v']),
        }

        print("Расчет завершен успешно.", flush=True)
        return result_data

    except Exception as e:
        logging.error(f"Processing failed: {e}")
        print(f"Error: {e}", flush=True)
