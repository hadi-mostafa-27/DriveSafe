import cv2
import mediapipe as mp
import dlib
import numpy as np
import os
import math
from collections import deque

# =======================
# YOLOv8 (Ultralytics)
# =======================
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO = None
    YOLO_AVAILABLE = False
    print("[WARN] YOLO not installed. Phone detection disabled.")


class DriverDistractionTracker:
    def __init__(self):
        # ---------- MediaPipe Face Mesh ----------
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            max_num_faces=1
        )

        # Eye indices
        self.MP_LEFT_EYE = [33, 160, 158, 133, 153, 144]
        self.MP_RIGHT_EYE = [362, 385, 387, 263, 373, 380]

        # ---------- Dlib for head pose ----------
        self.detector = dlib.get_frontal_face_detector()
        base_dir = os.path.dirname(os.path.abspath(__file__))
        predictor_path = os.path.join(base_dir, "models", "shape_predictor_68_face_landmarks.dat")

        if os.path.exists(predictor_path):
            self.predictor = dlib.shape_predictor(predictor_path)
        else:
            self.predictor = None
            print(f"[WARN] Predictor not found: {predictor_path}")

        # ---------- EAR / Blink ----------
        self.EAR_THRESHOLD = 0.19
        self.EAR_CONSEC_FRAMES = 4
        self.close_counter = 0

        # ---------- NEW STABLE HEAD POSE SETTINGS ----------
        self.HARD_TURN = 25  # Degrees: Only VERY strong turn triggers alert
        self.yaw_history = deque(maxlen=15)  # More smoothing = more stability

        # ---------- YOLO phone detection ----------
        self.yolo_model = None
        if YOLO_AVAILABLE:
            try:
                self.yolo_model = YOLO("yolov8n.pt")
                print("[INFO] YOLOv8n loaded successfully.")
            except Exception as e:
                print("[WARN] YOLO failed:", e)
                self.yolo_model = None

        self.yolo_frame_stride = 3
        self.yolo_frame_counter = 0
        self.last_phone_detected = False

    # =========================================================
    # ---------------------- EAR METHODS ----------------------
    # =========================================================
    def compute_EAR(self, pts):
        A = math.dist(pts[1], pts[5])
        B = math.dist(pts[2], pts[4])
        C = math.dist(pts[0], pts[3]) + 1e-6
        return (A + B) / (2.0 * C)

    def mediapipe_EAR(self, lm, frame):
        h, w = frame.shape[:2]

        def extract(idx_list):
            return [(int(lm.landmark[i].x * w), int(lm.landmark[i].y * h)) for i in idx_list]

        L = extract(self.MP_LEFT_EYE)
        R = extract(self.MP_RIGHT_EYE)
        return (self.compute_EAR(L) + self.compute_EAR(R)) / 2.0

    def dlib_EAR(self, frame, shape):
        L = [(shape.part(i).x, shape.part(i).y) for i in [36, 37, 38, 39, 40, 41]]
        R = [(shape.part(i).x, shape.part(i).y) for i in [42, 43, 44, 45, 46, 47]]
        return (self.compute_EAR(L) + self.compute_EAR(R)) / 2.0

    def is_eye_closed(self, m_ear, d_ear):
        ear = d_ear if d_ear is not None else m_ear
        if ear is None:
            self.close_counter = 0
            return False
        if ear < self.EAR_THRESHOLD:
            self.close_counter += 1
        else:
            self.close_counter = 0
        return self.close_counter >= self.EAR_CONSEC_FRAMES

    # =========================================================
    # -------------------- HEAD POSE  ------------------
    # =========================================================
    def get_head_pose(self, frame, shape):
        pts = np.array([
            (shape.part(30).x, shape.part(30).y),  # Nose
            (shape.part(8).x, shape.part(8).y),    # Chin
            (shape.part(36).x, shape.part(36).y),  # Left eye
            (shape.part(45).x, shape.part(45).y),  # Right eye
            (shape.part(48).x, shape.part(48).y),  # Mouth left
            (shape.part(54).x, shape.part(54).y)   # Mouth right
        ], dtype=float)

        model_pts = np.array([
            (0, 0, 0),
            (0, -63.6, -12.5),
            (-43.3, 32.7, -26),
            (43.3, 32.7, -26),
            (-28.9, -28.9, -24.1),
            (28.9, -28.9, -24.1)
        ])

        size = frame.shape
        focal = size[1]
        center = (size[1] / 2, size[0] / 2)
        cam = np.array([
            [focal, 0, center[0]],
            [0, focal, center[1]],
            [0, 0, 1]
        ])
        dist = np.zeros((4, 1))

        _, rv, _ = cv2.solvePnP(model_pts, pts, cam, dist)
        rmat, _ = cv2.Rodrigues(rv)
        euler, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)
        yaw = float(euler[1])
        pitch = float(euler[0])
        return yaw, pitch

    # =========================================================
    # ----------------- YOLO PHONE DETECTION ------------------
    # =========================================================
    def detect_phone(self, frame):
        """
        Runs YOLO every few frames and draws a red box + label around phones.
        Returns True if any phone is detected in this frame.
        """
        if self.yolo_model is None:
            return False

        phone_detected = False
        try:
            results = self.yolo_model(frame, imgsz=480, conf=0.35, verbose=False)
            for r in results:
                if not hasattr(r, "boxes") or r.boxes is None:
                    continue

                for box in r.boxes:
                    cls = int(box.cls[0])
                    label = self.yolo_model.names[cls]

                    # Match typical phone labels
                    if label.lower() in ["cell phone", "mobile phone", "phone"] or "phone" in label.lower():
                        phone_detected = True
                        x1, y1, x2, y2 = box.xyxy[0].tolist()
                        conf = float(box.conf[0])

                        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                        # Draw red rectangle around phone
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

                        # Label above box
                        cv2.putText(
                            frame,
                            f"PHONE {conf:.2f}",
                            (x1, max(20, y1 - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            (0, 0, 255),
                            2
                        )
            return phone_detected
        except Exception as e:
            print(f"[YOLO] Error during detection: {e}")
            return False

    # =========================================================
    # ---------------------- MAIN LOGIC -----------------------
    # =========================================================
    def process_frame(self, frame):
        """
        Processes a single frame:
        - YOLO phone detection (+ red box & label)
        - Head pose estimation
        - Eye state (OPEN/CLOSED)
        - Draws overlays for:
            * STATE (SAFE / DISTRACTED)
            * EYES: OPEN/CLOSED
            * HEAD: direction
            * PHONE: YES/NO
        Returns:
            state: "SAFE" or "DISTRACTED"
            info: dict with head/phone/eye_closed/yaw/pitch
        """
        # ---------------- YOLO Phone ----------------
        self.yolo_frame_counter = (self.yolo_frame_counter + 1) % self.yolo_frame_stride
        if self.yolo_frame_counter == 0:
            phone = self.detect_phone(frame)
            self.last_phone_detected = phone
        else:
            # Reuse last result between YOLO frames
            phone = self.last_phone_detected

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.detector(gray)

        yaw = 0.0
        pitch = None
        m_ear = None
        d_ear = None
        eye_closed = False

        shape = None

        # ---------- Dlib face + head pose ----------
        if faces and self.predictor:
            shape = self.predictor(gray, faces[0])
            try:
                raw_yaw, pitch = self.get_head_pose(frame, shape)
            except Exception as e:
                print(f"[HEAD] Error in pose estimation: {e}")
                raw_yaw = 0.0

            # ---------- SMOOTH YAW ----------
            self.yaw_history.append(raw_yaw)
            yaw = sum(self.yaw_history) / len(self.yaw_history)

        # ---------- HEAD CLASSIFICATION (ONLY HARD) ----------
        if abs(yaw) < 10:
            yaw = 0.0

        if yaw >= self.HARD_TURN:
            head = "Right-Hard"
        elif yaw <= -self.HARD_TURN:
            head = "Left-Hard"
        else:
            head = "Center"

        # ---------- Eyes using dlib (if available) ----------
        if faces and shape:
            try:
                d_ear = self.dlib_EAR(frame, shape)
            except Exception as e:
                print(f"[EAR-DLIB] Error: {e}")
                d_ear = None

        # ---------- Eyes using MediaPipe ----------
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb)

        if results.multi_face_landmarks:
            lm = results.multi_face_landmarks[0]
            try:
                m_ear = self.mediapipe_EAR(lm, frame)
            except Exception as e:
                print(f"[EAR-MP] Error: {e}")
                m_ear = None

        eye_closed = self.is_eye_closed(m_ear, d_ear)

        # ---------- Final STATE ----------
        if phone or eye_closed or "Hard" in head:
            state = "DISTRACTED"
        else:
            state = "SAFE"

        # =====================================================
        # --------------- DRAW OVERLAYS ON FRAME --------------
        # =====================================================
        h, w = frame.shape[:2]
        base_x = 20
        base_y = 40
        line_gap = 40

        # STATE
        state_color = (0, 255, 0) if state == "SAFE" else (0, 0, 255)
        cv2.putText(
            frame,
            f"STATE: {state}",
            (base_x, base_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            state_color,
            3
        )

        # EYES
        if eye_closed:
            eye_text = "EYES: CLOSED"
            eye_color = (0, 0, 255)
        else:
            eye_text = "EYES: OPEN"
            eye_color = (0, 255, 0)

        cv2.putText(
            frame,
            eye_text,
            (base_x, base_y + line_gap),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            eye_color,
            3
        )

        # HEAD
        cv2.putText(
            frame,
            f"HEAD: {head}",
            (base_x, base_y + 2 * line_gap),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (255, 255, 0),
            3
        )

        # PHONE
        phone_color = (0, 255, 0) if not phone else (0, 0, 255)
        cv2.putText(
            frame,
            f"PHONE: {'YES' if phone else 'NO'}",
            (base_x, base_y + 3 * line_gap),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            phone_color,
            3
        )

        # Optional: small yaw debug (in degrees) at bottom-left
        cv2.putText(
            frame,
            f"YAW: {yaw:.1f}",
            (base_x, h - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (200, 200, 200),
            2
        )

        # ---------- Return ----------
        return state, {
            "head": head,
            "phone": phone,
            "eye_closed": eye_closed,
            "yaw": yaw,
            "pitch": pitch
        }
