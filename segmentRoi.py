from pathlib import Path
import cv2
import numpy as np
from ultralytics import YOLO
import os


def roiSegmentation(image, save_path, model):
    # Predict text line segmentation on the input image
    res = model.predict(source=image, conf=0.25, classes = 2)
    for r in res:
        img = np.copy(r.orig_img)
        img_name = Path(r.path).stem

        # Sort contours based on y-coordinate in ascending order
        contours_sorted = sorted(r, key=lambda x: x.boxes.xyxy[0][1])

        for ci, c in enumerate(contours_sorted):
            label = c.names[c.boxes.cls.tolist().pop()]
            if c.masks is not None and c.masks.xy is not None:  # Check if masks are available
                contour = c.masks.xy.pop().astype(np.int32).reshape(-1, 1, 2)
                b_mask = np.zeros(img.shape[:2], np.uint8)
                _ = cv2.drawContours(b_mask, [contour], -1, (255, 255, 255), cv2.FILLED)
                mask3ch = cv2.cvtColor(b_mask, cv2.COLOR_GRAY2BGR)
                isolated = cv2.bitwise_and(mask3ch, img)

                x1, y1, x2, y2 = c.boxes.xyxy.cpu().numpy().squeeze().astype(np.int32)
                iso_crop = isolated[y1:y2, x1:x2]
                # Formulate the segmented file name based on y-coordinate
                segmented_file_name = f'{y1}.png'
                segmented_file_name = os.path.join(save_path, segmented_file_name)
                _ = cv2.imwrite(segmented_file_name, iso_crop)
            else:
                print("No contour masks found for", img_name, label)