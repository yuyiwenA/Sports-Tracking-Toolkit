"""Assign each tracked player to a team from jersey colour.

Almost every meaningful statement about a football match needs to know which
side a player is on. "A pass" is only a pass if the receiver is a teammate; if
he is not, the same observation is a turnover, which means the opposite thing.
Without a team label the event layer can only say "the ball moved from person 7
to person 12", which is not analysis.

The signal used here is the colour of the torso: players on one side wear the
same kit, and the two kits differ. This is the "similar yet distinguishable
appearance" property that the SportsMOT authors identify as the defining
characteristic of sports tracking, and it is available without any labels,
without a GPU, and without a network call.

Two implementation details matter for it to work on real footage:

* the torso is sampled instead of the whole box, because the box contains grass at
  the feet and often a second player's leg;
* pitch pixels are masked out by hue, because a green-dominant descriptor would
  cluster players by how much grass happens to be in their box.

When the clusters do not separate well the assigner reports ``reliable=False``
and labels everyone ``unknown``. A wrong team label silently inverts the meaning
of every downstream event, which is worse than admitting the colours could not
be separated.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from .trajectory_data import FrameTransform, TrackRow

#: OpenCV hue range (0-179) covering grass and most green backgrounds.
GRASS_HUE = (35, 90)


@dataclass
class TeamResult:
    """Team assignment plus the evidence behind it."""

    teams: Dict[int, str]                 # track_id -> "A" / "B" / "official" / "unknown"
    colours: Dict[str, Tuple[int, int, int]]  # label -> BGR
    reliable: bool
    silhouette: float
    n_clusters: int
    tracks_considered: int
    reason: str = ""

    def team_of(self, track_id: int) -> str:
        return self.teams.get(track_id, "unknown")

    @property
    def strength(self) -> str:
        """How much to trust the labels.

        A binary reliable flag hides the case that matters most: clusters that
        separate just enough to pass a threshold but not enough to act on. On
        the broadcast clip the red and white kits do separate, but at a
        silhouette of 0.18, which is enough to colour a diagram and not enough to
        let a turnover be inferred from it.
        """
        if not self.reliable:
            return "none"
        if self.silhouette >= 0.40:
            return "strong"
        if self.silhouette >= 0.25:
            return "moderate"
        return "weak"


# --------------------------------------------------------------------------- #
# descriptors
# --------------------------------------------------------------------------- #
def _torso_descriptor(frame: np.ndarray, box: Tuple[int, int, int, int]) -> Optional[np.ndarray]:
    """Hue-saturation histogram of the torso, with pitch pixels suppressed."""
    x1, y1, x2, y2 = box
    h, w = frame.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if x2 - x1 < 4 or y2 - y1 < 6:
        return None

    # torso only: skip the head/neck band and the legs
    ty1 = y1 + int(0.12 * (y2 - y1))
    ty2 = y1 + int(0.60 * (y2 - y1))
    tx1 = x1 + int(0.20 * (x2 - x1))
    tx2 = x1 + int(0.80 * (x2 - x1))
    patch = frame[ty1:ty2, tx1:tx2]
    if patch.size == 0:
        return None

    hsv = cv2.cvtColor(cv2.resize(patch, (32, 32), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2HSV)
    hue = hsv[:, :, 0]
    sat = hsv[:, :, 1].astype(np.float32)
    val = hsv[:, :, 2].astype(np.float32)

    weight = (sat / 255.0) * (val / 255.0)
    # suppress pitch: a saturated green pixel is background
    grass = (hue >= GRASS_HUE[0]) & (hue <= GRASS_HUE[1]) & (sat > 60)
    weight[grass] = 0.0
    # suppress blown-out highlights and deep shadow, which carry no hue
    weight[(val > 245) | (val < 25)] *= 0.15

    hist, _, _ = np.histogram2d(
        hue.ravel(), sat.ravel(), bins=(16, 8), range=((0, 180), (0, 256)), weights=weight.ravel()
    )
    total = hist.sum()
    if total <= 1e-6:
        return None
    return (hist / total).ravel().astype(np.float32)


def _mean_colour(frame: np.ndarray, box: Tuple[int, int, int, int]) -> Optional[Tuple[int, int, int]]:
    """Representative kit colour: the most saturated third of the torso pixels.

    Averaging every non-pitch pixel mixes the shirt with its own shadow, the
    highlights on it and whatever is behind the player, which turns a bright red
    kit into a muddy grey. The most saturated pixels are the ones carrying the
    kit's actual hue, so they are the ones worth reporting.
    """
    x1, y1, x2, y2 = box
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
    if x2 - x1 < 4 or y2 - y1 < 6:
        return None
    ty1 = y1 + int(0.12 * (y2 - y1))
    ty2 = y1 + int(0.60 * (y2 - y1))
    patch = frame[ty1:ty2, x1:x2]
    if patch.size == 0:
        return None
    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    mask = ~((hsv[:, :, 0] >= GRASS_HUE[0]) & (hsv[:, :, 0] <= GRASS_HUE[1]) & (hsv[:, :, 1] > 60))
    if mask.sum() < 8:
        return None
    pixels = patch[mask].reshape(-1, 3).astype(np.float32)
    saturation = hsv[:, :, 1][mask].reshape(-1)
    keep = max(1, int(0.35 * len(pixels)))
    order = np.argsort(saturation)[::-1][:keep]
    bgr = pixels[order].mean(axis=0)
    return int(bgr[0]), int(bgr[1]), int(bgr[2])


# --------------------------------------------------------------------------- #
# clustering
# --------------------------------------------------------------------------- #
def _kmeans(x: np.ndarray, k: int, iters: int = 60, seed: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    """Small k-means with k-means++ seeding (avoids a scikit-learn dependency)."""
    rng = np.random.default_rng(seed)
    n = len(x)
    k = max(1, min(k, n))
    centers = [x[rng.integers(n)]]
    for _ in range(k - 1):
        d2 = np.min(np.stack([((x - c) ** 2).sum(axis=1) for c in centers]), axis=0)
        total = d2.sum()
        idx = int(rng.choice(n, p=d2 / total)) if total > 0 else int(rng.integers(n))
        centers.append(x[idx])
    centers = np.asarray(centers, dtype=np.float32)

    labels = np.zeros(n, dtype=int)
    for _ in range(iters):
        dist = np.linalg.norm(x[:, None, :] - centers[None, :, :], axis=2)
        labels = dist.argmin(axis=1)
        new = np.stack(
            [x[labels == i].mean(axis=0) if np.any(labels == i) else centers[i] for i in range(k)]
        )
        if np.allclose(new, centers, atol=1e-5):
            break
        centers = new
    return labels, centers


def _silhouette(x: np.ndarray, labels: np.ndarray) -> float:
    """Mean silhouette coefficient; ``-1`` when a cluster is degenerate."""
    n = len(x)
    if n < 3:
        return -1.0
    dist = np.linalg.norm(x[:, None, :] - x[None, :, :], axis=2)
    scores = []
    for i in range(n):
        same = labels == labels[i]
        same[i] = False
        if not same.any():
            return -1.0
        a = dist[i, same].mean()
        b = min(
            dist[i, labels == c].mean()
            for c in np.unique(labels)
            if c != labels[i] and np.any(labels == c)
        )
        denom = max(a, b)
        scores.append((b - a) / denom if denom > 0 else 0.0)
    return float(np.mean(scores))


# --------------------------------------------------------------------------- #
# assigner
# --------------------------------------------------------------------------- #
class TeamAssigner:
    """Cluster player tracks by jersey colour.

    ``n_teams`` defaults to 2 because that is what a match has, and because
    letting the cluster count float is actively harmful here: on the broadcast
    clip an automatic search settled on three clusters and the labels then
    assigned the third, 24-player group to "official", filing a whole team as
    match officials. The domain gives the number for free, so it is used, and
    the silhouette score is reported so a weak separation is still visible.
    Set ``n_teams <= 0`` to search instead.
    """

    def __init__(
        self,
        n_teams: int = 2,
        samples_per_track: int = 5,
        min_track_frames: int = 6,
        max_teams: int = 4,
        min_silhouette: float = 0.12,
        official_max_share: float = 0.15,
    ) -> None:
        self.n_teams = int(n_teams)
        self.samples_per_track = int(samples_per_track)
        self.min_track_frames = int(min_track_frames)
        self.max_teams = int(max_teams)
        self.min_silhouette = float(min_silhouette)
        self.official_max_share = float(official_max_share)

    def fit(
        self,
        video_path: str,
        person_tracks: Dict[int, List[TrackRow]],
        transform: Optional[FrameTransform] = None,
        processed_size: Optional[Tuple[int, int]] = None,
    ) -> TeamResult:
        """Sample torso colours and cluster them into teams.

        ``transform`` undoes the pipeline's crop and resize so boxes land on the
        right pixels. Passing only ``processed_size`` (the old signature) applies
        a scale with no crop offset, which silently samples the wrong region
        whenever the run used ``--crop``.
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return TeamResult({}, {}, False, -1.0, 0, 0, f"cannot open {video_path}")

        native_w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        native_h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        if transform is None:
            # Old signature: scale only, no crop offset. Kept so existing callers
            # keep working, but any run that cropped will sample the wrong region.
            proc_w = int(processed_size[0]) if processed_size else int(native_w)
            proc_h = int(processed_size[1]) if processed_size else int(native_h)
            transform = FrameTransform(int(native_w), int(native_h), proc_w, proc_h, None)

        # one forward pass is far cheaper than hundreds of random seeks on a 4K
        # file: an h264
        # seek decodes from the previous keyframe, so skipping ahead with grab()
        # is dramatically cheaper than set(POS_FRAMES, ...) per sample
        wanted: Dict[int, List[Tuple[int, Tuple[int, int, int, int]]]] = {}
        for tid, rows in person_tracks.items():
            if len(rows) < self.min_track_frames:
                continue
            idxs = np.linspace(0, len(rows) - 1, min(self.samples_per_track, len(rows))).astype(int)
            for i in idxs:
                row = rows[int(i)]
                box = transform.box_to_source(row.bbox)
                wanted.setdefault(row.frame, []).append((tid, box))

        descriptors: Dict[int, List[np.ndarray]] = {}
        colours: Dict[int, List[Tuple[int, int, int]]] = {}
        position = 0
        for frame in sorted(wanted):
            if frame < position:
                continue
            while position < frame:
                if not cap.grab():
                    break
                position += 1
            ok, img = cap.read()
            if not ok:
                continue
            position = frame + 1
            for tid, box in wanted[frame]:
                desc = _torso_descriptor(img, box)
                if desc is not None:
                    descriptors.setdefault(tid, []).append(desc)
                    colour = _mean_colour(img, box)
                    if colour is not None:
                        colours.setdefault(tid, []).append(colour)
        cap.release()

        usable = {tid: np.mean(v, axis=0) for tid, v in descriptors.items()}
        if len(usable) < 2:
            return TeamResult(
                {tid: "unknown" for tid in person_tracks}, {}, False, -1.0, 0, len(usable),
                "not enough player crops with usable colour",
            )

        ids = list(usable)
        matrix = np.stack([usable[t] for t in ids])

        if self.n_teams > 0:
            k = max(2, min(self.n_teams, len(ids) - 1))
            labels, _ = _kmeans(matrix, k)
            score = _silhouette(matrix, labels)
        else:
            best = (-2.0, 2, None)
            for candidate_k in range(2, min(self.max_teams, len(ids) - 1) + 1):
                candidate_labels, _ = _kmeans(matrix, candidate_k)
                candidate_score = _silhouette(matrix, candidate_labels)
                if candidate_score > best[0]:
                    best = (candidate_score, candidate_k, candidate_labels)
            score, k, labels = best

        if labels is None or score < self.min_silhouette:
            return TeamResult(
                {tid: "unknown" for tid in person_tracks}, {}, False, float(score), k or 0, len(ids),
                f"colour clusters did not separate (silhouette {score:.3f} < {self.min_silhouette})",
            )

        # Rank the clusters by size: the two largest are the teams. A remaining
        # cluster counts as officials only if it is small. Treating
        # any third cluster as officials files a whole team under the wrong label.
        sizes = {c: int((labels == c).sum()) for c in np.unique(labels)}
        ranked = sorted(sizes, key=lambda c: -sizes[c])
        label_of = {}
        for position, cluster in enumerate(ranked):
            if position == 0:
                label_of[cluster] = "A"
            elif position == 1:
                label_of[cluster] = "B"
            elif sizes[cluster] <= self.official_max_share * len(ids):
                label_of[cluster] = "official"
            else:
                label_of[cluster] = f"cluster{position + 1}"

        teams: Dict[int, str] = {}
        cluster_colour: Dict[str, List[Tuple[int, int, int]]] = {}
        for tid, cluster in zip(ids, labels.tolist()):
            name = label_of[cluster]
            teams[tid] = name
            cluster_colour.setdefault(name, []).extend(colours.get(tid, []))
        for tid in person_tracks:
            teams.setdefault(tid, "unknown")

        final_colours = {
            name: tuple(int(v) for v in np.median(np.asarray(values), axis=0))
            for name, values in cluster_colour.items()
            if values
        }
        return TeamResult(
            teams=teams,
            colours=final_colours,
            reliable=True,
            silhouette=float(score),
            n_clusters=int(k),
            tracks_considered=len(ids),
            reason=f"{k} colour clusters, silhouette {score:.3f}",
        )


def team_colour_hex(bgr: Tuple[int, int, int]) -> str:
    b, g, r = bgr
    return f"#{r:02x}{g:02x}{b:02x}"
