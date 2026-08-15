from ultralytics import YOLO
import cv2

model = YOLO("yolo26n.pt", task="detect")

video_path = r"C:\Users\LAXMI SAI KUMAR\Downloads\Helmet Detection.mov"

cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Cannot open video")
    exit()

while True:

    ret, frame = cap.read()

    if not ret:
        break

    results = model(
        frame,
        classes=[3],       # Motorcycle only
        conf=0.15,
        imgsz=1280,
        verbose=False
    )

    annotated = results[0].plot()

    motorcycle_count = len(results[0].boxes)

    cv2.putText(
        annotated,
        f"Motorcycles detected: {motorcycle_count}",
        (20, 45),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 0, 255),
        3
    )

    cv2.imshow("Motorcycle Detection", annotated)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()