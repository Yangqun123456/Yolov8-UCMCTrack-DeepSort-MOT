import cv2
import numpy as np
from shapely.geometry import Point

# 检查一个边界框是否在区域内的函数
def is_inside_region(bbox, region):
    x1, y1, x2, y2 = bbox
    bbox_center = Point((x1 + x2) / 2, (y1 + y2) / 2)
    return region["polygon"].contains(bbox_center)

def showCounterText(img_box, region, line_thickness=2):
    # 在区域中心绘制计数
    region_label = str(region["counts"])
    centroid_x, centroid_y = map(int, region["polygon"].centroid.coords[0])

    text_size, _ = cv2.getTextSize(region_label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, thickness=line_thickness)
    text_x = centroid_x - text_size[0] // 2
    text_y = centroid_y + text_size[1] // 2

    cv2.rectangle(
        img_box,
        (text_x - 5, text_y - text_size[1] - 5),
        (text_x + text_size[0] + 5, text_y + 5),
        region["region_color"],
        -1,
    )
    cv2.putText(
        img_box, region_label, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, region["text_color"], line_thickness
    )
    cv2.polylines(img_box, [np.array(region["polygon"].exterior.coords, dtype=np.int32)], True, region["region_color"], line_thickness)