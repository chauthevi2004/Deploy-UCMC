import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from tracker.ucmc import UCMCTrack
from detector.mapper import Mapper
import tempfile
import os
from pathlib import Path

class Detection:
    def __init__(self, id, bb_left=0, bb_top=0, bb_width=0, bb_height=0, conf=0, det_class=0):
        self.id = id
        self.bb_left = bb_left
        self.bb_top = bb_top
        self.bb_width = bb_width
        self.bb_height = bb_height
        self.conf = conf
        self.det_class = det_class
        self.track_id = 0
        self.y = np.zeros((2, 1))
        self.R = np.eye(4)

    def __str__(self):
        return 'd{}, bb_box:[{},{},{},{}], conf={:.2f}, class{}, uv:[{:.0f},{:.0f}], mapped to:[{:.1f},{:.1f}]'.format(
            self.id, self.bb_left, self.bb_top, self.bb_width, self.bb_height, self.conf, self.det_class,
            self.bb_left + self.bb_width / 2, self.bb_top + self.bb_height, self.y[0, 0], self.y[1, 0])

class Detector:
    def __init__(self):
        self.seq_length = 0
        self.gmc = None

    def load(self, cam_para_file):
        self.mapper = Mapper(cam_para_file, "MOT17")
        # Thay thế YOLOv8x bằng YOLOv10n
        self.model = YOLO('pretrained/yolov10n.pt')

    def get_dets(self, img, conf_thresh=0, det_classes=[0]):
        dets = []
        frame = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.model(frame, imgsz=1088)

        det_id = 0
        for box in results[0].boxes:
            conf = box.conf.cpu().numpy()[0]
            bbox = box.xyxy.cpu().numpy()[0]
            cls_id = box.cls.cpu().numpy()[0]
            w = bbox[2] - bbox[0]
            h = bbox[3] - bbox[1]
            if w <= 10 and h <= 10 or cls_id not in det_classes or conf <= conf_thresh:
                continue

            det = Detection(det_id)
            det.bb_left = bbox[0]
            det.bb_top = bbox[1]
            det.bb_width = w
            det.bb_height = h
            det.conf = conf
            det.det_class = cls_id
            det.y, det.R = self.mapper.mapto([det.bb_left, det.bb_top, det.bb_width, det.bb_height])
            det_id += 1

            dets.append(det)

        return dets

def app():
    # Upload video file using Streamlit
    video_file = st.file_uploader("Tải lên một video", type=["mp4", "avi", "mov"])

    if video_file is not None:
        # Tạo thư mục lưu video nếu chưa có
        upload_dir = Path("uploaded_videos")
        upload_dir.mkdir(exist_ok=True)
        os.chmod(upload_dir, 0o755)

        # Lưu video vào thư mục
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_file:
            tmp_file.write(video_file.read())
            tmp_file_path = tmp_file.name

        # Hiển thị video gốc
        st.subheader("Video Gốc (Chưa Xử Lý)")
        st.video(video_file)  # Play the video using the uploaded bytes

        # Mở video bằng OpenCV
        cap = cv2.VideoCapture(tmp_file_path)

        # Kiểm tra xem video có mở thành công không
        if not cap.isOpened():
            st.error("Error: Video file could not be opened!")
            return

        # Lấy thông tin về độ phân giải video
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        st.write(f"Video resolution: {width}x{height}")

        # Thêm nút để bắt đầu quá trình theo dõi
        if st.button('Start Tracking'):
            # Khởi tạo detector và tracker
            detector = Detector()
            detector.load("demo/cam_para.txt")
            
            tracker = UCMCTrack(100.0, 100.0, 5, 5, 10, 30.0, cap.get(cv2.CAP_PROP_FPS), "MOT", 0.5, False, None)

            # Tạo danh sách lưu trữ các frame đã xử lý
            processed_frames = []

            # Tạo placeholder để hiển thị trạng thái xử lý
            status_placeholder = st.empty()
            status_placeholder.text("Đang xử lý video...")

            # Lấy tổng số frame trong video
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            # Tạo thanh tiến trình để hiển thị trạng thái xử lý
            progress_bar = st.progress(0)  # Khởi tạo thanh tiến trình

            # Xử lý từng frame trong video
            frame_id = 1
            while True:
                ret, frame = cap.read()
                if not ret:
                    break  # Nếu không đọc được frame, thoát khỏi vòng lặp

                # Lấy kết quả phát hiện đối tượng
                dets = detector.get_dets(frame, conf_thresh=0.01)

                # Cập nhật tracker với các đối tượng đã phát hiện
                tracker.update(dets, frame_id)

                # Vẽ bounding box và ID đối tượng
                for det in dets:
                    if det.track_id > 0:
                        cv2.rectangle(frame, (int(det.bb_left), int(det.bb_top)),
                                    (int(det.bb_left + det.bb_width), int(det.bb_top + det.bb_height)),
                                    (0, 255, 0), 2)
                        cv2.putText(frame, str(det.track_id), (int(det.bb_left), int(det.bb_top)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                # Thêm frame đã xử lý vào danh sách
                processed_frames.append(frame)

                # Cập nhật thanh tiến trình theo tiến độ
                progress = int((frame_id / total_frames) * 100)
                progress_bar.progress(progress)  # Cập nhật thanh tiến trình

                frame_id += 1

            cap.release()

            # Sau khi hoàn thành, cập nhật trạng thái và hiển thị video đã xử lý
            status_placeholder.text("Đã hoàn thành xử lý!")

            # Tạo video xử lý lưu vào bộ nhớ
            video_out_path = "processed_video.mp4"
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(video_out_path, fourcc, 30.0, (width, height))

            # Ghi các frame đã xử lý vào file video
            for frame in processed_frames:
                out.write(frame)
            out.release()

            # Hiển thị video đã xử lý sau khi tracking
            st.subheader("Video Đã Xử Lý")
            with open(video_out_path, "rb") as f:
                st.video(f.read())

    else:
        st.warning("Please upload a video.")

if __name__ == "__main__":
    app()
