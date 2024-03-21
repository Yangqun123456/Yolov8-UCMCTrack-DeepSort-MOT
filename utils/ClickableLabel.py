# This file is part of Yolov8-UCMCTrack-DeepSort-MOT which is released under the AGPL-3.0 license.
# See file LICENSE or go to https://github.com/Yangqun123456/Yolov8-UCMCTrack-DeepSort-MOT/tree/main/LICENSE for full license details.

from PySide6.QtWidgets import QLabel
from PySide6.QtCore import Qt
# 测距
selected_boxes = {}
distence_boxes = []
distence_trk_ids = []
left_mouse_count = 0
# 测速
reg_pts = [(20, 400), (1260, 400)]


class ClickableLabel(QLabel):
    def __init__(self, parent=None):
        super(ClickableLabel, self).__init__(parent)
        self.iw = 0
        self.ih = 0
        self.reg_pts = []
        self.drawing_line = False
        # 功能bool变量
        self.distence_estimate = False
        self.speed_estimate = False
        global left_mouse_count
        left_mouse_count = 0

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            x, y = self.getPixelLocation(event)
            if self.distence_estimate:
                global left_mouse_count
                left_mouse_count += 1
                for box, track_id in zip(distence_boxes, distence_trk_ids):
                    if box[0] < x < box[2] and box[1] < y < box[3] and track_id not in selected_boxes:
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
                self.reg_pts.append((int(x), int(y)))

        if event.button() == Qt.RightButton:
            if self.distence_estimate:
                set_selected_boxes(None, {})
                left_mouse_count = 0

    def mouseReleaseEvent(self, event):
        if self.speed_estimate and self.drawing_line:
            x, y = self.getPixelLocation(event)
            self.reg_pts.append((int(x), int(y)))
            global reg_pts
            reg_pts = self.reg_pts
            self.drawing_line = False

    def getPixelLocation(self, event):
        local_point = self.mapFromGlobal(event.globalPos())
        # Calculate the scale factors
        scale_x = self.iw / self.pixmap().width() if self.pixmap().width() > 0 else 0
        scale_y = self.ih / self.pixmap().height() if self.pixmap().height() > 0 else 0
        # Convert the coordinates from points to pixels
        pixel_x = local_point.x() * scale_x
        pixel_y = local_point.y() * scale_y
        return pixel_x, pixel_y


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
