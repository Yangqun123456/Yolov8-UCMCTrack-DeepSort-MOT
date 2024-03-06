from ultralytics import YOLO
from UCMCTracker.tracker.ucmc import UCMCTrack
from UCMCTracker.detector.mapper import Mapper
import argparse
import cv2
import numpy as np

parser = argparse.ArgumentParser(description='Process some arguments.')
parser.add_argument('--cam_para', type=str,
                    default="UCMCTracker/demo/cam_para.txt", help='camera parameter file name')
parser.add_argument('--wx', type=float, default=5, help='wx')
parser.add_argument('--wy', type=float, default=5, help='wy')
parser.add_argument('--vmax', type=float, default=10, help='vmax')
parser.add_argument('--a', type=float, default=100.0,
                    help='assignment threshold')
parser.add_argument('--cdt', type=float, default=10.0,
                    help='coasted deletion time')
parser.add_argument('--high_score', type=float,
                    default=0.5, help='high score threshold')
parser.add_argument('--conf_thresh', type=float, default=0.01,
                    help='detection confidence threshold')
args = parser.parse_args()


# 定义一个Detection类，包含id,bb_left,bb_top,bb_width,bb_height,conf,det_class
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
            self.bb_left+self.bb_width/2, self.bb_top+self.bb_height, self.y[0, 0], self.y[1, 0])

    def __repr__(self):
        return self.__str__()


# Detector类，用于从Yolo检测器获取目标检测的结果
class Detector:
    def __init__(self):
        self.seq_length = 0
        self.gmc = None

    def load(self, model, cam_para_file):
        self.mapper = Mapper(cam_para_file, "MOT17")
        self.model = YOLO(model)

    def get_dets(self, img, det_classes=[0], conf_thresh=0.01):

        dets = []

        # 将帧从 BGR 转换为 RGB（因为 OpenCV 使用 BGR 格式）
        frame = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # 使用 RTDETR 进行推理
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

            # 新建一个Detection对象
            det = Detection(det_id)
            det.bb_left = bbox[0]
            det.bb_top = bbox[1]
            det.bb_width = w
            det.bb_height = h
            det.conf = conf
            det.det_class = cls_id
            det.y, det.R = self.mapper.mapto(
                [det.bb_left, det.bb_top, det.bb_width, det.bb_height])
            det_id += 1

            dets.append(det)

        return dets

    def count_targets_and_classes(self, dets):
        target_count = len(dets)
        class_count = len(set(det.det_class for det in dets))
        return target_count, class_count


class UCMCTracker:
    def __init__(self, model, fps, args=args):
        self.detector = Detector()
        self.detector.load(model, args.cam_para)
        self.tracker = UCMCTrack(args.a, args.a, args.wx, args.wy, args.vmax,
                                 args.cdt, fps, "MOT", args.high_score, False, None)
