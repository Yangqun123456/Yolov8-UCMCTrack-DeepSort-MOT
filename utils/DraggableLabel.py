# This file is part of Yolov8-UCMCTrack-DeepSort-MOT which is released under the AGPL-3.0 license.
# See file LICENSE or go to https://github.com/Yangqun123456/Yolov8-UCMCTrack-DeepSort-MOT/tree/main/LICENSE for full license details.

import random
from PySide6.QtWidgets import QLabel
from PySide6.QtCore import Qt
from shapely.geometry import Point, Polygon

# 区域计数
counting_regions = [
    {
        "name": "YOLOv8 Polygon Region",
        # Polygon points
        "polygon": Polygon([(50, 80), (250, 20), (450, 80), (400, 350), (100, 350)]),
        "counts": 0,
        "dragging": False,
        "region_color": (255, 42, 4),  # BGR Value
        "text_color": (255, 255, 255),  # Region Text Color
    },
    {
        "name": "YOLOv8 Rectangle Region",
        # Polygon points
        "polygon": Polygon([(200, 250), (440, 250), (440, 550), (200, 550)]),
        "counts": 0,
        "dragging": False,
        "region_color": (37, 255, 225),  # BGR Value
        "text_color": (0, 0, 0),  # Region Text Color
    },
]
# 越线计数
line = [(100, 500), (300, 800)]
# 测距
selected_boxes = {}
distence_boxes = []
distence_trk_ids = []
left_mouse_count = 0
# 测速
reg_pts = [(20, 400), (1260, 400)]


class DraggableLabel(QLabel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMouseTracking(True)
        self.counting_regions = counting_regions
        self.current_region = None
        self.ih = 0
        self.iw = 0
        self.drawing_line = False
        self.line = []
        self.reg_pts = []
        # 功能bool值
        self.crossing_line = False
        self.region_counter = False
        self.distence_estimate = False
        self.speed_estimate = False
        global left_mouse_count
        left_mouse_count = 0

    def getPixelLocation(self, event):
        local_point = self.mapFromGlobal(event.globalPos())
        # Calculate the scale factors
        scale_x = self.iw / self.pixmap().width() if self.pixmap().width() > 0 else 0
        scale_y = self.ih / self.pixmap().height() if self.pixmap().height() > 0 else 0
        # Convert the coordinates from points to pixels
        pixel_x = local_point.x() * scale_x
        pixel_y = local_point.y() * scale_y
        return pixel_x, pixel_y

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            pixel_x, pixel_y = self.getPixelLocation(event)
            if self.region_counter:
                for region in self.counting_regions:
                    if region["polygon"].contains(Point((pixel_x, pixel_y))):
                        self.current_region = region
                        self.current_region["dragging"] = True
                        self.current_region["offset_x"] = pixel_x
                        self.current_region["offset_y"] = pixel_y
                        break
            elif self.crossing_line:
                self.drawing_line = True
                self.line = []
                self.line.append((int(pixel_x), int(pixel_y)))
            elif self.distence_estimate:
                global left_mouse_count
                left_mouse_count += 1
                for box, track_id in zip(distence_boxes, distence_trk_ids):
                    if box[0] < pixel_x < box[2] and box[1] < pixel_y < box[3] and track_id not in selected_boxes:
                        if left_mouse_count <= 2:
                            set_selected_boxes(track_id, [])
                            set_selected_boxes(track_id, box)
                        else:
                            set_selected_boxes(None, [])
                            set_selected_boxes(track_id, [])
                            set_selected_boxes(track_id, box)
                            left_mouse_count = 1
            elif self.speed_estimate:
                self.drawing_line = True
                self.reg_pts = []
                self.reg_pts.append((int(pixel_x), int(pixel_y)))

        if event.button() == Qt.RightButton:
            if self.distence_estimate:
                set_selected_boxes(None, {})
                left_mouse_count = 0

    def mouseMoveEvent(self, event):
        if self.current_region is not None and self.current_region["dragging"]:
            pixel_x, pixel_y = self.getPixelLocation(event)
            dx = pixel_x - self.current_region["offset_x"]
            dy = pixel_y - self.current_region["offset_y"]
            self.current_region["polygon"] = Polygon(
                [(p[0] + dx, p[1] + dy)
                 for p in self.current_region["polygon"].exterior.coords]
            )
            self.current_region["offset_x"] = pixel_x
            self.current_region["offset_y"] = pixel_y
            self.update()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            if self.current_region is not None and self.current_region["dragging"]:
                self.current_region["dragging"] = False
        if self.crossing_line and self.drawing_line:
            pixel_x, pixel_y = self.getPixelLocation(event)
            self.line.append((int(pixel_x), int(pixel_y)))
            global line
            line = self.line
            self.drawing_line = False
        elif self.speed_estimate and self.drawing_line:
            x, y = self.getPixelLocation(event)
            self.reg_pts.append((int(x), int(y)))
            global reg_pts
            reg_pts = self.reg_pts
            self.drawing_line = False


def get_line():
    return line


def set_distence_boxes(boxes, trk_ids):
    global distence_boxes
    global distence_trk_ids
    distence_boxes = boxes
    distence_trk_ids = trk_ids


def set_selected_boxes(track_id, box):
    global selected_boxes
    if track_id is None:
        selected_boxes = {}
    elif box != {}:  # if box is not empty
        selected_boxes[track_id] = box
    else:
        selected_boxes = {}


def get_selected_boxes():
    global selected_boxes
    return selected_boxes


def get_reg_pts():
    global reg_pts
    return reg_pts


# def set_counting_regions(region):
#     global counting_regions
#     counting_regions = []
#     # 将输入的字符串转换为点的列表
#     points = list(map(int, region.split(',')))
#     points = [(points[i], points[i+1]) for i in range(0, len(points), 2)]
#     counting_regions.append({
#         "name": "YOLOv8 Polygon Region",
#         # Polygon points
#         "polygon": Polygon(points),
#         "counts": 0,
#         "dragging": False,
#         "region_color": (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)),  # BGR Value
#         "text_color": (255, 255, 255),  # Region Text Color
#     })