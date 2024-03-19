from PySide6.QtCore import Signal, QObject
import numpy as np
import time
import cv2
import datetime
import os
import torch
from ultralytics.data.loaders import LoadStreams
from ultralytics.engine.predictor import BasePredictor
from ultralytics.solutions import distance_calculation, heatmap, speed_estimation
from ultralytics.utils.torch_utils import smart_inference_mode
from utils.draw_img import draw_boxes, is_integer_string
from tracker.UCMCTracker.ucmcTracker import UCMCTracker
from tracker.DeepSortTracker.deepsort import DeepSortTracker
from utils.chart import Scatter, analyzeData
from utils.region_counter import is_inside_region, showCounterText
from utils.DraggableLabel import counting_regions, get_line

video_id_count = 0


class YoloPredictor(BasePredictor, QObject):
    yolo2main_box_img = Signal(np.ndarray)  # 绘制了标签与锚框的图像的信号
    yolo2main_second_img = Signal(np.ndarray)  # 绘制热力图的信号
    yolo2main_status_msg = Signal(str)  # 检测/暂停/停止/测试完成等信号
    yolo2main_fps = Signal(str)  # fps
    yolo2main_progress = Signal(int)  # 进度条
    yolo2main_class_num = Signal(int)  # 当前帧类别数
    yolo2main_target_num = Signal(int)  # 当前帧目标数

    def __init__(self):
        super(YoloPredictor, self).__init__()
        QObject.__init__(self)

        # GUI args
        self.used_model_name = None      # 使用过的检测模型名称
        self.new_model_name = None       # 新更改的模型
        self.source = ''                 # 输入源str
        self.stop_dtc = False            # 终止bool
        self.continue_dtc = True         # 暂停bool
        self.save_res = True            # 保存MP4
        self.conf_thres = 0.01           # conf
        self.progress_value = 0          # 进度条的值
        self.lock_id = None              # 锁定的ID
        self.tracker = 'UCMCTracker'     # 跟踪器
        self.region_counter = False      # 区域计数
        self.crossing_line = False     # 跨线计数
        self.show_hot_img = False        # 显示热力图
        self.show_speed_img = False      # 显示速度估计
        self.show_distence_img = False   # 显示距离估计

    # 单目标跟踪
    def single_object_tracking(self, dets, img_box, org, store_xyxy_for_id):
        for det in dets:
            store_xyxy_for_id[det.track_id] = [
                det.bb_left, det.bb_top, det.bb_left + det.bb_width, det.bb_top + det.bb_height]
            mask = np.zeros_like(img_box)
        try:
            x1, y1, x2, y2 = int(store_xyxy_for_id[self.lock_id][0]), int(store_xyxy_for_id[self.lock_id][1]), int(
                store_xyxy_for_id[self.lock_id][2]), int(store_xyxy_for_id[self.lock_id][3])
            cv2.rectangle(mask, (x1, y1), (x2, y2), (255, 255, 255), -1)
            result_mask = cv2.bitwise_and(org, mask)
            result_cropped = result_mask[y1:y2, x1:x2]
            height, width = result_cropped.shape[:2]
            result_cropped = cv2.resize(result_cropped, (width*3, height*3))
            return result_cropped

        except:
            cv2.destroyAllWindows()
            pass

    # 点击开始检测按钮后的检测事件
    @smart_inference_mode()
    def run(self):
        # try:
        LoadStreams.capture = None

        global video_id_count

        self.yolo2main_status_msg.emit('正在加载模型...')

        # 检查保存路径
        if self.save_res:
            if not os.path.exists('output'):
                os.mkdir('output')

        count = 0                       # 拿来参与算FPS的计数变量
        start_time = time.time()        # 拿来算FPS的计数变量
        if self.continue_dtc:  # 暂停与继续的切换
            try:
                video_out.release()
                video_id_count += 1
            except:
                pass
            if self.used_model_name != self.new_model_name:
                self.setup_model(self.new_model_name)
                self.used_model_name = self.new_model_name
            self.yolo2main_status_msg.emit('检测中...')
            # 绘图参数
            timesListGraph = []
            graphDataList = []
            # 获取视频流
            if is_integer_string(self.source):
                self.source = int(self.source)
            cap = cv2.VideoCapture(self.source)
            total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
            height, width = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(
                cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            if self.save_res:
                video_out = cv2.VideoWriter(
                    f'output/output_{video_id_count}.mp4', cv2.VideoWriter_fourcc(*'mp4v'), cap.get(cv2.CAP_PROP_FPS), (width, height))

            store_xyxy_for_id = {}
            class_list = [0, 1, 2, 5, 7, 8]
            ucmcTracker = UCMCTracker(
                self.new_model_name, cap.get(cv2.CAP_PROP_FPS))
            deepsortTracker = DeepSortTracker(self.new_model_name)

            # Init heatmap
            heatmap_obj = heatmap.Heatmap()
            heatmap_obj.set_args(colormap=cv2.COLORMAP_PARULA,
                                 imw=cap.get(cv2.CAP_PROP_FRAME_WIDTH),
                                 imh=cap.get(cv2.CAP_PROP_FRAME_HEIGHT),
                                 shape="circle")

            # Init speed-estimation obj
            speed_obj = speed_estimation.SpeedEstimator()
            speed_obj.set_args(reg_pts=[(0, 360), (1280, 360)],
                               names=ucmcTracker.detector.model.names)
            
            # Init distance-calculation obj
            dist_obj = distance_calculation.DistanceCalculation()
            dist_obj.set_args(names=ucmcTracker.detector.model.names)

            # 循环读取视频帧
            frame_id = 1
            while True:
                try:
                    if self.continue_dtc:
                        ret, img_box = cap.read()
                        img_second = np.copy(img_box)  # 用于生成热力图
                        org = np.copy(img_box)
                        # 初始化区域计数器
                        for region in counting_regions:
                            region["counts"] = 0
                        if not ret:
                            break
                        if self.tracker == 'UCMCTracker':
                            dets = ucmcTracker.detector.get_dets(
                                img_box, class_list, self.conf_thres)
                            ucmcTracker.tracker.update(dets, frame_id)
                        elif self.tracker == 'DeepSort':
                            dets = deepsortTracker.detector.get_dets(
                                img_box, class_list, self.conf_thres)
                            # 将 dets 转换为 DeepSort 所需的格式
                            bbox_xywh = []
                            confidences = []
                            cls_ids = []
                            for det in dets:
                                bbox_xywh.append(
                                    [det.bb_left, det.bb_top, det.bb_width, det.bb_height])
                                confidences.append(det.conf)
                                cls_ids.append(det.det_class)
                            bbox_xywh = torch.Tensor(bbox_xywh)
                            confidences = torch.Tensor(confidences)
                            # 使用 DeepSort 的 update 函数更新跟踪器的状态
                            outputs = deepsortTracker.tracker.update(
                                bbox_xywh, confidences, cls_ids, org)
                            # 将检测到的目标 ID 赋值给 det.track_id，并更新 bbox 值
                            for output, det in zip(outputs, dets):
                                x1, y1, x2, y2, track_id, track_oid = output
                                det.bb_left = x1
                                det.bb_top = y1
                                det.bb_width = x2 - x1
                                det.bb_height = y2 - y1
                                det.track_id = track_id
                                det.det_class = track_oid

                        # 热力图
                        if self.show_hot_img:
                            img_second = heatmap_obj.generate_heatmap(
                                img_second, dets)
                        # 速度估计
                        elif self.show_speed_img:
                            img_second = speed_obj.estimate_speed(img_second, dets)
                        # 距离估计
                        elif self.show_distence_img:
                            img_second = dist_obj.start_process(img_second, dets)

                        bbox_xyxy = []
                        identities = []
                        object_ids = []
                        for det in dets:
                            if det.track_id > 0:
                                # 计算bbox的坐标
                                bbox = [int(det.bb_left), int(det.bb_top), int(
                                    det.bb_left+det.bb_width), int(det.bb_top+det.bb_height)]
                                bbox_xyxy.append(bbox)
                                # 获取对象的ID
                                identities.append(det.track_id)
                                # 获取对象的类别ID
                                object_ids.append(det.det_class)
                                if self.region_counter:
                                    # Check if detection inside region
                                    for region in counting_regions:
                                        if is_inside_region(bbox, region):
                                            region["counts"] += 1

                        # 使用draw_boxes函数进行绘制
                        img_box = draw_boxes(
                            img_box, bbox_xyxy, ucmcTracker.detector.model.names, object_ids, identities, get_line(), self.crossing_line)

                        # 区域计数
                        if self.region_counter:
                            for region in counting_regions:
                                showCounterText(img_box, region)

                        frame_id += 1
                        target_num, class_num = ucmcTracker.detector.count_targets_and_classes(
                            dets)

                        video_out.write(img_box)

                        if self.stop_dtc:
                            video_out.release()
                            video_id_count += 1
                            self.source = None
                            self.yolo2main_status_msg.emit('检测终止')
                            LoadStreams.capture = 'release'  # 这里是为了终止使用摄像头检测函数的线程，改了yolo源码
                            break

                        try:
                            # 绘图
                            if frame_id % 20 == 0:
                                graph_data = analyzeData(object_ids)
                                graphDataList.append(graph_data)
                                timesListGraph.append(
                                    datetime.datetime.now().strftime('%H:%M:%S'))
                                Scatter(timesListGraph, graphDataList)

                            # 显示结果视频
                            self.yolo2main_box_img.emit(img_box)
                            if self.show_hot_img or self.show_speed_img or self.show_distence_img:
                                self.yolo2main_second_img.emit(img_second)

                            # 进度条
                            try:
                                self.progress_value = int(  # 进度条
                                    (frame_id/total_frames)*1000)
                                self.yolo2main_progress.emit(
                                    self.progress_value)
                            except:
                                pass

                            # 单目标跟踪
                            if self.lock_id is not None:
                                self.lock_id = int(self.lock_id)
                                try:
                                    result_cropped = self.single_object_tracking(
                                        dets, img_box, org, store_xyxy_for_id)
                                    cv2.imshow(
                                        f'OBJECT-ID:{self.lock_id}', result_cropped)
                                    cv2.moveWindow(
                                        f'OBJECT-ID:{self.lock_id}', 0, 0)
                                    # press esc to quit
                                    if cv2.waitKey(5) & 0xFF == 27:
                                        self.lock_id = None
                                        cv2.destroyAllWindows()
                                except Exception as e:
                                    cv2.destroyAllWindows()
                                    pass

                            self.yolo2main_class_num.emit(class_num)
                            self.yolo2main_target_num.emit(target_num)
                        except:
                            pass
                        count += 1

                        if count % 3 == 0 and count >= 3:  # 计算FPS
                            self.yolo2main_fps.emit(
                                str(int(3/(time.time()-start_time))))
                            start_time = time.time()

                    if self.stop_dtc:
                        video_out.release()
                        video_id_count += 1
                        self.source = None
                        self.yolo2main_status_msg.emit('检测终止')
                        break

            # 检测截止（本地文件检测）
                except StopIteration:
                    video_out.release()
                    video_id_count += 1
                    self.yolo2main_status_msg.emit('检测完成')
                    self.yolo2main_progress.emit(1000)
                    cv2.destroyAllWindows()
                    break
            try:
                video_out.release()
                video_id_count += 1
            except:
                pass
