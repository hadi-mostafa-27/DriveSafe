# pylint:disable=no-member
import sys
import time
import csv
import os
import cv2
import paho.mqtt.client as mqtt

from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QHBoxLayout, QVBoxLayout,
    QGroupBox, QLabel, QPushButton
)
    # If you need more widgets later, import them here.
from PyQt5.QtGui import QPixmap, QImage, QFont

from driver_tracker_YOLO import DriverDistractionTracker

# MQTT Settings
MQTT_BROKER = "broker.emqx.io"
MQTT_PORT = 1883
MQTT_TOPIC_ALERT = "drivesafe/alerts/hadi"


class DriverMonitorWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("DriveSafe – Simplified UI")
        self.resize(1200, 800)

        central = QWidget()
        self.setCentralWidget(central)
        layout = QHBoxLayout(central)

        # =====================================================
        # LEFT PANEL (STATUS + CONTROLS)
        # =====================================================
        left = QVBoxLayout()
        layout.addLayout(left)

        # ---------- Live Status ----------
        stats = QGroupBox("Driver Status")
        s = QVBoxLayout()

        self.state_lbl = QLabel("Status: Not Started")
        self.state_lbl.setFont(QFont("Arial", 14, QFont.Bold))

        self.eye_lbl = QLabel("Eye: ---")
        self.head_lbl = QLabel("Head: ---")
        self.phone_lbl = QLabel("Phone: ---")

        self.eye_lbl.setFont(QFont("Arial", 12))
        self.head_lbl.setFont(QFont("Arial", 12))
        self.phone_lbl.setFont(QFont("Arial", 12))

        s.addWidget(self.state_lbl)
        s.addWidget(self.eye_lbl)
        s.addWidget(self.head_lbl)
        s.addWidget(self.phone_lbl)
        stats.setLayout(s)
        left.addWidget(stats)

        # ---------- Start/Stop Buttons ----------
        btns = QVBoxLayout()
        self.start_btn = QPushButton("START MONITORING")
        self.stop_btn = QPushButton("STOP")
        self.stop_btn.setEnabled(False)

        self.start_btn.setMinimumHeight(50)
        self.stop_btn.setMinimumHeight(50)

        btns.addWidget(self.start_btn)
        btns.addWidget(self.stop_btn)
        left.addLayout(btns)

        # =====================================================
        # RIGHT PANEL (VIDEO FEED)
        # =====================================================
        vid_box = QGroupBox("Camera")
        v2 = QVBoxLayout()
        self.video_lbl = QLabel()
        self.video_lbl.setAlignment(Qt.AlignCenter)
        self.video_lbl.setMinimumSize(900, 600)
        v2.addWidget(self.video_lbl)
        vid_box.setLayout(v2)
        layout.addWidget(vid_box)

        # =====================================================
        # SYSTEM SETUP
        # =====================================================
        self.tracker = DriverDistractionTracker()
        self.cap = None
        self.timer = QTimer()
        self.timer.timeout.connect(self.process_frame)

        self.last_alert_time = 0
        self.alert_cooldown = 3  # seconds

        self.init_mqtt()

        self.start_btn.clicked.connect(self.start_session)
        self.stop_btn.clicked.connect(self.stop_session)

    # MQTT Setup
    def init_mqtt(self):
        try:
            self.mqtt_client = mqtt.Client(client_id="DriveSafe_PC")
            self.mqtt_client.connect(MQTT_BROKER, MQTT_PORT, 60)
            self.mqtt_client.loop_start()
            print("[MQTT] Connected.")
        except Exception as e:
            print(f"[MQTT] Error: {e}")
            self.mqtt_client = None

    def send_mqtt(self, msg):
        if self.mqtt_client:
            try:
                self.mqtt_client.publish(MQTT_TOPIC_ALERT, msg, qos=0)
                print(f"[MQTT] Sent: {msg}")
            except Exception as e:
                print(f"[MQTT] Publish error: {e}")

    # START
    def start_session(self):
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            print("[ERROR] Cannot open webcam.")
            return

        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)

        self.timer.start(33)  # ~30 FPS

    # STOP
    def stop_session(self):
        if self.cap:
            self.cap.release()
            self.cap = None
        self.timer.stop()
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)

        self.state_lbl.setText("Status: Stopped")
        self.eye_lbl.setText("Eye: ---")
        self.head_lbl.setText("Head: ---")
        self.phone_lbl.setText("Phone: ---")

    # FRAME LOOP
    def process_frame(self):
        if not self.cap:
            return

        ok, frame = self.cap.read()
        if not ok:
            return

        frame = cv2.flip(frame, 1)
        state, info = self.tracker.process_frame(frame)

        eye = info["eye_closed"]
        head = info["head"]
        phone = info["phone"]

        # Update left-panel labels
        self.eye_lbl.setText(f"Eye: {'CLOSED' if eye else 'OPEN'}")
        self.head_lbl.setText(f"Head: {head}")
        self.phone_lbl.setText(f"Phone: {'DETECTED' if phone else '---'}")
        self.state_lbl.setText(f"Status: {state}")

        now = time.time()
        alert = None

        # Priority Alerts (PHONE > EYE > HEAD)
        if now - self.last_alert_time > self.alert_cooldown:
            if phone:
                alert = "PHONE"
            elif eye:
                alert = "EYE"
            elif head in ["Left-Hard", "Right-Hard"]:
                alert = "HEAD"

            if alert:
                self.send_mqtt(alert)
                self.send_mqtt("LED_RED")
                self.last_alert_time = now
        else:
            # While in cooldown, keep green LED (no new alert)
            self.send_mqtt("LED_GREEN")

        # Display frame (already has overlays from tracker)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        q = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
        self.video_lbl.setPixmap(QPixmap.fromImage(q))


def main():
    app = QApplication(sys.argv)
    win = DriverMonitorWindow()
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
