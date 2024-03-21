# This file is part of Yolov8-UCMCTrack-DeepSort-MOT which is released under the AGPL-3.0 license.
# See file LICENSE or go to https://github.com/Yangqun123456/Yolov8-UCMCTrack-DeepSort-MOT/tree/main/LICENSE for full license details.

from PySide6.QtWidgets import QLabel
from PySide6.QtCore import Qt
from shapely.geometry import Point, Polygon

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

line = [(100, 500), (300, 800)]


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
        # 功能bool值
        self.crossing_line = False
        self.region_counter = False

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
        if self.drawing_line:
            pixel_x, pixel_y = self.getPixelLocation(event)
            self.line.append((int(pixel_x), int(pixel_y)))
            global line
            line = self.line
            self.drawing_line = False


def get_line():
    return line
