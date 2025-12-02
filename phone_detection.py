import cv2
from ultralytics import YOLO

def main():
    # Load YOLOv8n model (will auto-download first time)
    model = YOLO("yolov8n.pt")
    print("[INFO] YOLOv8n model loaded.")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Cannot open webcam.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Failed to grab frame.")
            break

        # Run YOLO on the frame
        results = model(frame, imgsz=480, conf=0.35, verbose=False)

        # Draw only phone detections
        for r in results:
            if not hasattr(r, "boxes") or r.boxes is None:
                continue

            for box in r.boxes:
                cls = int(box.cls[0])
                label = model.names[cls]

                # Only highlight phones
                if label.lower() in ["cell phone", "mobile phone", "phone"]:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    conf = float(box.conf[0])

                    # Draw rectangle
                    cv2.rectangle(
                        frame,
                        (int(x1), int(y1)),
                        (int(x2), int(y2)),
                        (0, 0, 255),
                        2
                    )

                    text = f"{label} {conf:.2f}"
                    cv2.putText(
                        frame, text,
                        (int(x1), int(y1) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 0, 255),
                        2
                    )

        cv2.imshow("Phone Detection (YOLOv8n)", frame)

        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Webcam closed. Bye.")

if __name__ == "__main__":
    main()
