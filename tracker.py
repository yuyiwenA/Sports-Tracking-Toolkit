from ultralytics.trackers.byte_tracker import BYTETracker
import numpy as np

class Tracker:
    def __init__(self, track_thresh=0.25, track_buffer=30, match_thresh=0.8):
        args = type('Args', (object,), {
            "track_thresh": track_thresh,
            "track_buffer": track_buffer,
            "match_thresh": match_thresh,
            "mot20": False
        })
        self.tracker = BYTETracker(args, frame_rate=25)

    def update(self, dets, img_shape):
        if len(dets)==0:
            return []
        det_array = np.array(dets, dtype=np.float32)
        online_targets = self.tracker.update(det_array, img_shape, img_shape)
        tracks = []
        for t in online_targets:
            x1,y1,x2,y2 = t.tlbr
            tid = t.track_id
            tracks.append([int(x1),int(y1),int(x2),int(y2),tid])
        return tracks
