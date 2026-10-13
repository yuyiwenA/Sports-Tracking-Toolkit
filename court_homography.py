import cv2
import numpy as np

class CourtHomography:
    def __init__(self):
        # 标准篮球场 4角真实坐标 (单位米)
        self.real_court = np.array([
            [0,0],
            [28,0],
            [28,15],
            [0,15]
        ], dtype=np.float32)
        self.H = None

    def calibrate_manual(self, image_points):
        img_pts = np.array(image_points, dtype=np.float32)
        self.H, _ = cv2.findHomography(img_pts, self.real_court)

    def pixel2court(self, px, py):
        if self.H is None:
            return [px,py]
        pt = np.array([[px,py]], dtype=np.float32)
        res = cv2.perspectiveTransform(pt, self.H)
        return res[0][0].tolist()
