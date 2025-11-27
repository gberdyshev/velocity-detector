from ml.data import colors
from ultralytics import YOLO
import cv2
import numpy as np
import os

model = YOLO("./ml/best-3.pt")


def process_image(image_path) -> list:
    objects = list()
    image = cv2.imread(image_path)
    results_list = model(image)
    results = results_list[0]
    
    classes_names = results.names
    classes = results.boxes.cls.cpu().numpy()
    boxes = results.boxes.xyxy.cpu().numpy().astype(np.int32)
    for class_id, box in zip(classes, boxes):
        class_name = classes_names[int(class_id)]
        color = colors[int(class_id) % len(colors)]
        
        x1, y1, x2, y2 = box
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            image, class_name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2
        )

        do = {
            "x1": int(x1), 
            "y1": int(y1), 
            "x2": int(x2), 
            "y2": int(y2), 
            "obj_type": class_name
        }
        objects.append(do)

    new_image_path = (
        os.path.splitext(image_path)[0] + "_yolo" + os.path.splitext(image_path)[1]
    )
    cv2.imwrite(new_image_path, image)
    
    return objects