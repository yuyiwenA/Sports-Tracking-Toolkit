from ultralytics import YOLO

class Detector:
    def __init__(self, model_path, conf_thres=0.3):
        self.model = YOLO(model_path)
        self.conf_thres = conf_thres

    def detect(self, frame):
        res = self.model(frame, conf=self.conf_thres)[0]
        boxes = res.boxes
        dets = []
        if boxes is not None:
            for box in boxes:
                x1,y1,x2,y2 = map(int,box.xyxy[0])
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                dets.append([x1,y1,x2,y2,conf,cls])
        return dets
