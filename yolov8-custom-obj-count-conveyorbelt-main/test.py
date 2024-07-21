import cv2
import pandas as pd
from ultralytics import YOLO
from tracker import Tracker
import cvzone

# Load your YOLO model
model = YOLO('best.pt')

def RGB(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:  
        colorsBGR = [x, y]
        print(colorsBGR)

cv2.namedWindow('RGB')
cv2.setMouseCallback('RGB', RGB)

# Open the video file
cap = cv2.VideoCapture('9716407-hd_1920_1080_25fps.mp4')

# Load the class list
with open("coco1.txt", "r") as my_file:
    class_list = my_file.read().split("\n")

count = 0

# Define the counting area (x, y, width, height)
counting_area = (90, 179, 357, 300)  # Lower and wider area

# Initialize the Tracker with the counting area
tracker = Tracker(counting_area)

# Product count
product_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    count += 1
    if count % 3 != 0:
        continue

    frame = cv2.resize(frame, (1020, 500))

    results = model.predict(frame)
    a = results[0].boxes.data
    px = pd.DataFrame(a).astype("float")

    list_of_boxes = []

    for index, row in px.iterrows():
        x1 = int(row[0])
        y1 = int(row[1])
        x2 = int(row[2])
        y2 = int(row[3])
        d = int(row[5])
        c = class_list[d]

        list_of_boxes.append([x1, y1, x2, y2])

    bbox_idx = tracker.update(list_of_boxes)
    
    for bbox in bbox_idx:
        x3, y3, x4, y4, id = bbox
        cx = (x3 + x4) // 2
        cy = (y3 + y4) // 2
        
        cv2.circle(frame, (cx, cy), 4, (255, 0, 0), -1)
        cv2.rectangle(frame, (x3, y3), (x4, y4), (255, 0, 255), 2)
        cvzone.putTextRect(frame, f'{id}', (cx, cy), 1, 1)
        
        if tracker.is_within_area([x3, y3, x4, y4]) and id not in tracker.counted_ids:
            tracker.counted_ids.add(id)
            product_count += 1
            print(f"Product {id} counted. Total count: {product_count}")

    # Draw the counting area
    area_x, area_y, area_w, area_h = counting_area
    cv2.rectangle(frame, (area_x, area_y), (area_x + area_w, area_y + area_h), (0, 255, 0), 2)

    # Display the product count in the top-right corner
    cvzone.putTextRect(frame, f'Count: {product_count}', (frame.shape[1] - 200, 50), 2, 2, offset=10, border=2)

    cv2.imshow("RGB", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
