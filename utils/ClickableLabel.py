from PySide6.QtWidgets import QLabel
from PySide6.QtCore import Qt

selected_boxes = {}
distence_boxes = []
distence_trk_ids = []
left_mouse_count = 0


class ClickableLabel(QLabel):
    def __init__(self, parent=None):
        super(ClickableLabel, self).__init__(parent)
        self.iw = 0
        self.ih = 0
        global left_mouse_count
        left_mouse_count = 0

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            global left_mouse_count
            left_mouse_count += 1
            x, y = self.getPixelLocation(event)
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

        if event.button() == Qt.RightButton:
            set_selected_boxes(None, {})
            left_mouse_count = 0

    def getPixelLocation(self, event):
        local_point = self.mapFromGlobal(event.globalPos())
        # Calculate the scale factors
        scale_x = self.iw / self.pixmap().width()
        scale_y = self.ih / self.pixmap().height()
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
