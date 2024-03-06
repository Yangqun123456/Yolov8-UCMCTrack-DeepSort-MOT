import cv2
from ultralytics import YOLO
from tracker.DeepSortTracker.utils.parser import get_config
from tracker.DeepSortTracker.deep_sort import DeepSort

cfg_deep = get_config()
cfg_deep.merge_from_file("tracker/DeepSortTracker/configs/deep_sort.yaml")

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

    def __str__(self):
        return 'd{}, bb_box:[{},{},{},{}], conf={:.2f}, class{}, uv:[{:.0f},{:.0f}], mapped to:[{:.1f},{:.1f}]'.format(
            self.id, self.bb_left, self.bb_top, self.bb_width, self.bb_height, self.conf, self.det_class,
            self.bb_left+self.bb_width/2, self.bb_top+self.bb_height, self.y[0, 0], self.y[1, 0])

    def __repr__(self):
        return self.__str__()


# Detector类，用于从Yolo检测器获取目标检测的结果
class Detector:
    def __init__(self):
        self.model = None

    def load(self, model):
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
            det_id += 1

            dets.append(det)

        return dets

    def count_targets_and_classes(self, dets):
        target_count = len(dets)
        class_count = len(set(det.det_class for det in dets))
        return target_count, class_count


class DeepSortTracker:
    def __init__(self, model):
        self.detector = Detector()
        self.detector.load(model)
        self.tracker= DeepSort(cfg_deep.DEEPSORT.REID_CKPT,
                            max_dist=cfg_deep.DEEPSORT.MAX_DIST, min_confidence=cfg_deep.DEEPSORT.MIN_CONFIDENCE,
                            nms_max_overlap=cfg_deep.DEEPSORT.NMS_MAX_OVERLAP, max_iou_distance=cfg_deep.DEEPSORT.MAX_IOU_DISTANCE,
                            max_age=cfg_deep.DEEPSORT.MAX_AGE, n_init=cfg_deep.DEEPSORT.N_INIT, nn_budget=cfg_deep.DEEPSORT.NN_BUDGET,
                            use_cuda=True)
