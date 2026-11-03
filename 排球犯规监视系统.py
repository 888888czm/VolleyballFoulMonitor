# -*- coding: utf-8 -*-
import sys
import os
import time
import cv2
import random
import traceback
import numpy as np
import pandas as pd
from collections import deque
from datetime import datetime
from dataclasses import dataclass
from typing import List, Dict, Optional

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QComboBox, QSpinBox, QTableWidget,
    QTableWidgetItem, QFileDialog, QMessageBox
)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QThread
from PyQt5.QtGui import QImage, QPixmap

from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

# ==========================================
# 全局异常拦截
# ==========================================
def excepthook(exc_type, exc_value, exc_tb):
    error_msg = "".join(traceback.format_exception(exc_type, exc_value, exc_tb))
    with open("error_log.txt", "w", encoding="utf-8") as f:
        f.write(error_msg)
    print(error_msg)
sys.excepthook = excepthook

def resource_path(relative_path):
    if hasattr(sys, '_MEIPASS'):
        return os.path.join(sys._MEIPASS, relative_path)
    return os.path.join(os.path.abspath("."), relative_path)

# ==========================================
# 数据与规则引擎
# ==========================================
@dataclass
class PlayerPose:
    track_id: int
    team: int
    keypoints: Dict[str, tuple]
    foot_x: float
    foot_y: float
    is_back_row: bool
    in_front_zone: bool

@dataclass
class FoulEvent:
    timestamp: float
    foul_type: str
    reason: str
    team: int
    player_id: Optional[int] = None
    confidence: float = 0.0
    clip_path: str = ""

class VolleyballFoulEngine:
    def __init__(self, net_y=0.45, center_line_x=0.5):
        self.net_y = net_y
        self.center_line_x = center_line_x
        self._last_foul_time = {}

    def evaluate(self, poses, frame_shape):
        h, w = frame_shape[:2]
        events = []
        for p in poses:
            for kp_name in ("left_wrist", "right_wrist", "left_shoulder", "right_shoulder"):
                kp = p.keypoints.get(kp_name)
                if kp is None: continue
                if abs(kp[0]/w - self.center_line_x) < 0.03 and abs(kp[1]/h - self.net_y) < 0.05:
                    ev = self._make_event(p, "触网", "击球动作中或干扰比赛时触及标志杆之间的球网")
                    if ev: events.append(ev)
            if abs(p.foot_x/w - self.center_line_x) < 0.02:
                ev = self._make_event(p, "过中线", "整只脚、手或身体越过中线并接触对方场区")
                if ev: events.append(ev)
        return events

    def _make_event(self, p, foul_type, reason, confidence=0.7):
        now = time.time()
        key = (p.track_id, foul_type)
        if now - self._last_foul_time.get(key, 0) < 5.0: return None # 同一人同种犯规5秒冷却
        self._last_foul_time[key] = now
        return FoulEvent(now, foul_type, reason, p.team, p.track_id, confidence)

class SimulatedDetector:
    """模拟检测器：周期性生成靠近中线的动作触发犯规"""
    def __init__(self, num=6): self.n = num
    def detect(self, frame, frame_index):
        h, w = frame.shape[:2]
        poses = []
        for team in (0, 1):
            for i in range(self.n):
                base_x = w * (0.15 + 0.7 * (i / max(self.n - 1, 1)))
                base_y = h * (0.3 + 0.4 * random.random())
                if team == 1: base_x = w - base_x
                # 周期性制造犯规
                if (frame_index // 150) % 3 == team and i == frame_index % self.n:
                    base_x = w * (0.5 + random.uniform(-0.02, 0.02))
                    base_y = h * (0.45 + random.uniform(-0.02, 0.02))
                kp = {
                    "left_wrist": (base_x + w*0.02, base_y - h*0.10), "right_wrist": (base_x - w*0.02, base_y - h*0.10),
                    "left_shoulder": (base_x + w*0.01, base_y - h*0.15), "right_shoulder": (base_x - w*0.01, base_y - h*0.15),
                }
                poses.append(PlayerPose(team*100+i, team, kp, base_x, base_y, i>=3, base_y < h*0.5))
        return poses

# ==========================================
# 报告生成器
# ==========================================
class ReportGenerator:
    def __init__(self, output_dir="reports"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        font_path = resource_path("fonts/simhei.ttf")
        if os.path.exists(font_path):
            pdfmetrics.registerFont(TTFont("SimHei", font_path))
            self.font_name = "SimHei"
        else:
            self.font_name = "Helvetica"

    def export_csv(self, events):
        if not events: return None
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.join(self.output_dir, f"foul_log_{ts}.csv")
        rows = [{"时间": datetime.fromtimestamp(e.timestamp).strftime("%H:%M:%S"), "队伍": "A" if e.team==0 else "B", 
                 "类型": e.foul_type, "原因": e.reason} for e in events]
        pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8-sig")
        return path

    def export_pdf(self, events):
        if not events: return None
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.join(self.output_dir, f"foul_report_{ts}.pdf")
        doc = SimpleDocTemplate(path, pagesize=A4)
        styles = getSampleStyleSheet()
        title_style = ParagraphStyle("T", parent=styles["Heading1"], fontName=self.font_name, fontSize=16)
        story = [Paragraph("排球比赛犯规监测报告", title_style), Spacer(1, 12)]
        data = [["时间", "队伍", "犯规类型", "原因摘要"]]
        for e in events:
            data.append([datetime.fromtimestamp(e.timestamp).strftime("%H:%M:%S"), "A" if e.team==0 else "B", e.foul_type, e.reason[:50]])
        table = Table(data, colWidths=[70, 40, 90, 280])
        table.setStyle(TableStyle([("FONTNAME", (0,0), (-1,-1), self.font_name), ("GRID", (0,0), (-1,-1), 0.5, colors.grey)]))
        story.append(table)
        doc.build(story)
        return path

# ==========================================
# 线程（包含自动录像功能）
# ==========================================
class DetectionWorker(QThread):
    foul_detected = pyqtSignal(object)
    frame_ready = pyqtSignal(object)

    def __init__(self, source, detector, engine, clip_dir="clips"):
        super(DetectionWorker, self).__init__()
        self.source = source
        self.detector = detector
        self.engine = engine
        self.clip_dir = clip_dir
        os.makedirs(clip_dir, exist_ok=True)
        self.running = False
        self._frame_buffer = deque(maxlen=75) # 保存前3秒画面
        self._recording_until = 0
        self._writer = None

    def run(self):
        try:
            cap = cv2.VideoCapture(self.source, cv2.CAP_DSHOW)
            if not cap.isOpened():
                self.frame_ready.emit(None)
                return

            self.running = True
            idx = 0
            while self.running:
                ret, frame = cap.read()
                if not ret: break
                idx += 1
                self._frame_buffer.append(frame.copy())
                
                if idx % 3 == 0:
                    self.frame_ready.emit(frame.copy())

                poses = self.detector.detect(frame, idx)
                events = self.engine.evaluate(poses, frame.shape)
                for ev in events:
                    self._start_recording(ev)
                    self.foul_detected.emit(ev)
                    
                self._write_frame(frame)
                time.sleep(0.03)

            cap.release()
        except Exception as e:
            print("线程崩溃:", e)
        finally:
            self._close_writer()

    def _start_recording(self, event):
        self._recording_until = time.time() + 3.0 # 录制犯规后3秒
        ts = datetime.now().strftime("%H%M%S")
        path = os.path.join(self.clip_dir, f"foul_{ts}_{event.foul_type}.mp4")
        event.clip_path = path
        if self._frame_buffer:
            h, w = self._frame_buffer[0].shape[:2]
            try:
                # 尝试初始化录像，如果缺少编码器会在这里报错，但不会闪退
                self._writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), 25, (w, h))
                for f in self._frame_buffer: 
                    self._writer.write(f)
            except Exception as e:
                print(f"[警告] 录像启动失败: {e}")

    def _write_frame(self, frame):
        if self._writer is not None and self._writer.isOpened():
            self._writer.write(frame)
            if time.time() > self._recording_until:
                self._close_writer()

    def _close_writer(self):
        if self._writer is not None:
            self._writer.release()
            self._writer = None

    def stop(self):
        self.running = False
        self.wait()

# ==========================================
# 主界面（恢复左右分栏与设定时间）
# ==========================================
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("排球犯规自动监测系统")
        self.resize(1100, 700)
        self.engine = VolleyballFoulEngine()
        self.reporter = ReportGenerator()
        self.events = []
        self.worker = None
        self._build_ui()

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)

        # 左侧
        left = QVBoxLayout()
        self.preview = QLabel("摄像头预览区域")
        self.preview.setAlignment(Qt.AlignCenter)
        self.preview.setMinimumSize(640, 360)
        self.preview.setStyleSheet("background:#111;color:#aaa;")
        left.addWidget(self.preview)

        ctrl = QHBoxLayout()
        self.source_combo = QComboBox()
        self.source_combo.addItems(["0（默认摄像头）", "1（外接USB摄像头）"])
        ctrl.addWidget(QLabel("视频源："))
        ctrl.addWidget(self.source_combo)
        
        ctrl.addWidget(QLabel("监测时长(分)："))
        self.duration_spin = QSpinBox()
        self.duration_spin.setRange(1, 300)
        self.duration_spin.setValue(30)
        ctrl.addWidget(self.duration_spin)

        self.start_btn = QPushButton("开始监测")
        self.start_btn.clicked.connect(self._start)
        self.stop_btn = QPushButton("停止")
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self._stop)
        ctrl.addWidget(self.start_btn)
        ctrl.addWidget(self.stop_btn)
        left.addLayout(ctrl)

        # 右侧
        right = QVBoxLayout()
        right.addWidget(QLabel("犯规事件日志"))
        self.table = QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(["时间", "队伍", "犯规类型", "原因"])
        self.table.horizontalHeader().setStretchLastSection(True)
        right.addWidget(self.table)

        btn_row = QHBoxLayout()
        self.csv_btn = QPushButton("导出 CSV")
        self.csv_btn.clicked.connect(self._export_csv)
        self.pdf_btn = QPushButton("导出 PDF")
        self.pdf_btn.clicked.connect(self._export_pdf)
        btn_row.addWidget(self.csv_btn)
        btn_row.addWidget(self.pdf_btn)
        right.addLayout(btn_row)

        root.addLayout(left, 3)
        root.addLayout(right, 2)

    def _start(self):
        source = int(self.source_combo.currentText()[0])
        self.worker = DetectionWorker(source, SimulatedDetector(), self.engine)
        self.worker.foul_detected.connect(self._on_foul)
        self.worker.frame_ready.connect(self._update_preview)
        self.worker.start()
        
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.statusBar().showMessage("监测已启动...")

        # 自动停止定时器
        minutes = self.duration_spin.value()
        QTimer.singleShot(minutes * 60 * 1000, self._stop)

    def _stop(self):
        if self.worker:
            self.worker.stop()
            self.worker = None
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.statusBar().showMessage("监测已停止。")

    def _update_preview(self, frame):
        if frame is None:
            self.preview.setText("无法读取摄像头")
            self._stop()
            return
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        img = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
        self.preview.setPixmap(QPixmap.fromImage(img).scaled(self.preview.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def _on_foul(self, event):
        self.events.append(event)
        row = self.table.rowCount()
        self.table.insertRow(row)
        self.table.setItem(row, 0, QTableWidgetItem(datetime.fromtimestamp(event.timestamp).strftime("%H:%M:%S")))
        self.table.setItem(row, 1, QTableWidgetItem("A" if event.team == 0 else "B"))
        self.table.setItem(row, 2, QTableWidgetItem(event.foul_type))
        self.table.setItem(row, 3, QTableWidgetItem(event.reason))
        self.statusBar().showMessage(f"报警：{event.foul_type}", 3000)

    def _export_csv(self):
        if not self.events: return QMessageBox.information(self, "提示", "暂无犯规事件。")
        path = self.reporter.export_csv(self.events)
        QMessageBox.information(self, "完成", f"CSV 已保存：\n{path}")

    def _export_pdf(self):
        if not self.events: return QMessageBox.information(self, "提示", "暂无犯规事件。")
        path = self.reporter.export_pdf(self.events)
        QMessageBox.information(self, "完成", f"PDF 已保存：\n{path}")

    def closeEvent(self, event):
        self._stop()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())