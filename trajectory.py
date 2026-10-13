import numpy as np
from filterpy.kalman import KalmanFilter

class TrajectoryAnalyzer:
    def __init__(self, fps=25):
        self.fps = fps
        self.dt = 1.0 / fps
        self.track_history = dict()

    def add_point(self, tid, cx, cy):
        if tid not in self.track_history:
            kf = KalmanFilter(dim_x=4, dim_z=2)
            kf.F = np.array([[1,0,self.dt,0],
                             [0,1,0,self.dt],
                             [0,0,1,0],
                             [0,0,0,1]])
            kf.H = np.array([[1,0,0,0],
                             [0,1,0,0]])
            kf.P *= 1000
            kf.R *= 1
            kf.x = np.array([cx, cy, 0,0]).reshape(4,1)
            self.track_history[tid] = {"kf":kf, "points":[]}
        item = self.track_history[tid]
        item["kf"].predict()
        item["kf"].update([cx, cy])
        smooth_x = float(item["kf"].x[0])
        smooth_y = float(item["kf"].x[1])
        item["points"].append([smooth_x, smooth_y])
        return smooth_x, smooth_y

    def calc_speed(self, tid):
        pts = self.track_history[tid]["points"]
        if len(pts)<2:
            return 0.0
        dx = pts[-1][0] - pts[-2][0]
        dy = pts[-1][1] - pts[-2][1]
        dist = np.hypot(dx, dy)
        speed = dist / self.dt
        return speed
