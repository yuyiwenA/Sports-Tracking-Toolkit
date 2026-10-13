import argparse
import cv2
import csv
from src.detector import Detector
from src.tracker import Tracker
from src.court_homography import CourtHomography
from src.trajectory import TrajectoryAnalyzer
from src.visualizer import Visualizer

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--model", default="yolov8n.pt")
    parser.add_argument("--conf", type=float, default=0.3)
    parser.add_argument("--save_csv", action="store_true", default=True)
    parser.add_argument("--save_video", action="store_true", default=True)
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    detector = Detector(args.model, args.conf)
    tracker = Tracker()
    traj_analyzer = TrajectoryAnalyzer(fps=fps)
    court = CourtHomography()
    # ===== 这里填入视频里球场四个角像素坐标，你可以后续改成自动检测 =====
    # 示例：court.calibrate_manual([[10,20],[900,10],[920,500],[5,480]])

    out_video = None
    if args.save_video:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out_video = cv2.VideoWriter("outputs/result.mp4", fourcc, fps, (w,h))

    csv_f = None
    csv_w = None
    if args.save_csv:
        csv_f = open("outputs/trajectory.csv","w",newline="",encoding="utf-8")
        csv_w = csv.writer(csv_f)
        csv_w.writerow(["frame","track_id","px","py","court_x","court_y","speed"])

    frame_id = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        dets = detector.detect(frame)
        tracks = tracker.update(dets, (h,w))
        frame = Visualizer.draw(frame, tracks)

        for t in tracks:
            x1,y1,x2,y2,tid = t
            cx = (x1+x2)/2
            cy = (y1+y2)/2
            sx, sy = traj_analyzer.add_point(tid, cx, cy)
            spd = traj_analyzer.calc_speed(tid)
            cx_court, cy_court = court.pixel2court(sx, sy)
            if args.save_csv:
                csv_w.writerow([frame_id, tid, sx, sy, cx_court, cy_court, spd])

        if out_video:
            out_video.write(frame)
        cv2.imshow("preview", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        frame_id += 1

    cap.release()
    if out_video: out_video.release()
    if csv_f: csv_f.close()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    import os
    os.makedirs("outputs", exist_ok=True)
    main()
