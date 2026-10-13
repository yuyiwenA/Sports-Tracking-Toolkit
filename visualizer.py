import cv2

class Visualizer:
    @staticmethod
    def draw(frame, tracks):
        for t in tracks:
            x1,y1,x2,y2,tid = t
            cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
            cv2.putText(frame, f"ID:{tid}", (x1,y1-5),
                        cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,255,0),2)
        return frame
