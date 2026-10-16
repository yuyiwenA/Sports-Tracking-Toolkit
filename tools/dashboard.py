"""Build a self-contained interactive replay dashboard from the analysis outputs.

The pipeline produces a video, a CSV and a few static charts. None of them can
show the ten seconds around a particular pass, which is what an analyst wants to
look at. This tool writes one HTML file that scrubs through the tracked
positions with the events marked.

Two constraints shape it:

* no external requests. No CDN, no web fonts, no charting library. The file
  renders offline, from a USB stick or on GitHub Pages. This costs some polish
  and removes a whole class of failure.
* a schematic instead of video. The map shows tracked positions rather than the
  footage, because embedding 20 MB of video as base64 would make the page
  unusable and the geometry is the subject.

Usage::

    python tools/dashboard.py --trajectory outputs/trajectory.csv --out outputs
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.ball_path import build_ball_path  # noqa: E402
from src.events import EventDetector  # noqa: E402
from src.selfcheck import SelfCheck  # noqa: E402
from src.teams import TeamAssigner, team_colour_hex  # noqa: E402
from src.trajectory_data import (  # noqa: E402
    load_report,
    load_trajectory,
    sibling_report,
    transform_from_report,
)

TEAM_COLOURS = {
    "A": "#2f6fed", "B": "#e0533d", "official": "#c9821a",
    "cluster3": "#12a37a", "cluster4": "#8a5cd6", "unknown": "#9aa4b0",
}

EVENT_COLOURS = {
    "pass": "#2f6fed",
    "turnover": "#e0533d",
    "sprint": "#12a37a",
    "change_of_direction": "#c9821a",
    "ball_speed_peak": "#8a5cd6",
    "shot": "#d0453a",
}

EVENT_LABELS = {
    "pass": "传球",
    "turnover": "球权转换",
    "sprint": "冲刺",
    "change_of_direction": "变向",
    "ball_speed_peak": "球速峰值",
    "shot": "射门",
}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python tools/dashboard.py",
        description="Write a self-contained interactive replay dashboard.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--trajectory", default="outputs/trajectory.csv")
    p.add_argument("--video", default=None, help="source video, for jersey-colour team assignment")
    p.add_argument("--out", default=None, help="output directory (default: alongside the CSV)")
    p.add_argument("--title", default=None, help="dashboard title (default: the video's name)")
    p.add_argument("--no-teams", action="store_true")
    p.add_argument("--teams", type=int, default=2,
                   help="number of teams to cluster for; <=0 searches for the best k instead")
    return p


def collect(csv_path: str, video: Optional[str], no_teams: bool, n_teams: int = 2):
    report = load_report(sibling_report(csv_path))
    video_meta = report.get("video", {})
    data = load_trajectory(csv_path, fps=video_meta.get("fps"))
    ball_path, _ = build_ball_path(data.ball_tracks(), fps=data.fps)

    teams = None
    if video and not no_teams and os.path.exists(video):
        import cv2

        cap = cv2.VideoCapture(video)
        src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        transform = transform_from_report(report, src_w, src_h)
        teams = TeamAssigner(n_teams=n_teams).fit(video, data.person_tracks(), transform=transform)

    detector = EventDetector(data, ball_path, teams=teams)
    events, segments, _ = detector.detect()
    scorecard = SelfCheck(data, ball_path).report()

    width, height = data.frame_size()
    if video_meta.get("frame_width"):
        width, height = video_meta["frame_width"], video_meta["frame_height"]

    players = []
    for row in data.person_rows:
        players.append([
            row.frame,
            row.track_id,
            round(row.x, 1),
            round(row.y, 1),
            team_index(teams, row.track_id),
        ])

    ball = [[int(f), round(float(p[0]), 1), round(float(p[1]), 1),
             bool(ball_path.interpolated[i])]
            for i, (f, p) in enumerate(zip(ball_path.frames, ball_path.xy))] if ball_path else []

    event_rows = [
        {
            "frame": e.frame,
            "t": round(e.time_s, 2),
            "kind": e.kind,
            "label": EVENT_LABELS.get(e.kind, e.kind),
            "colour": EVENT_COLOURS.get(e.kind, "#9aa4b0"),
            "track": e.track_id,
            "partner": e.partner_id,
            "team": e.team,
            "magnitude": e.magnitude,
            "confidence": e.confidence,
            "detail": e.detail,
        }
        for e in events
    ]

    possess = [
        {"track": s.track_id, "team": s.team, "start": s.start, "end": s.end}
        for s in segments
    ]

    payload = {
        "meta": {
            "source": report.get("source") or os.path.basename(csv_path),
            "fps": round(data.fps, 3),
            "width": int(width),
            "height": int(height),
            "first_frame": data.first_frame,
            "last_frame": data.last_frame,
            "calibration": (report.get("calibration") or {}).get("mode", "none"),
            "tracker": (report.get("tracker") or {}).get("used", "?"),
        },
        "teamColours": team_colours(teams),
        "teamLabels": sorted({p[4] for p in players}),
        "players": players,
        "ball": ball,
        "events": event_rows,
        "possession": possess,
        "scorecard": {
            "overall": scorecard["overall_score"],
            "grade": scorecard["overall_grade"],
            "metrics": scorecard["metrics"],
        },
        "eventColours": EVENT_COLOURS,
        "eventLabels": EVENT_LABELS,
    }
    return payload


def team_index(teams, track_id: int) -> int:
    """Index into the JS colour array; 5 is the 'unknown' slot."""
    if teams is None:
        return 5
    label = teams.team_of(track_id)
    return {"A": 0, "B": 1, "official": 2, "cluster3": 3, "cluster4": 4}.get(label, 5)


def team_colours(teams) -> Dict[str, str]:
    out = dict(TEAM_COLOURS)
    if teams is not None and teams.colours:
        for name, bgr in teams.colours.items():
            out[name] = team_colour_hex(bgr)
    return out


HTML = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title>
<style>
:root{
  --ink:#1d232b; --muted:#5b6673; --faint:#8b95a2;
  --line:#e6eaef; --line2:#d3dae1; --panel:#f8fafc; --accent:#2f6fed;
}
*{box-sizing:border-box}
body{margin:0;background:#fff;color:var(--ink);
  font-family:"Segoe UI","Helvetica Neue",Arial,"Microsoft YaHei","PingFang SC",sans-serif;font-size:14px;line-height:1.6}
.wrap{max-width:1120px;margin:0 auto;padding:32px 24px 72px}
h1{font-size:22px;font-weight:600;margin:0 0 4px;letter-spacing:-.01em}
.sub{color:var(--muted);font-size:13px}
.meta{display:flex;flex-wrap:wrap;gap:18px;margin:16px 0 22px;padding:12px 16px;
  background:var(--panel);border:1px solid var(--line)}
.meta div{font-size:12px;color:var(--muted)}
.meta b{display:block;font-size:15px;font-weight:600;color:var(--ink);margin-top:2px}
.grid{display:grid;grid-template-columns:minmax(0,1fr) 300px;gap:22px;align-items:start}
@media (max-width:900px){.grid{grid-template-columns:1fr}}
canvas{width:100%;display:block;background:#fbfcfd;border:1px solid var(--line);border-radius:3px}
.ctrl{display:flex;align-items:center;gap:12px;margin-top:12px;flex-wrap:wrap}
button{font:inherit;font-size:13px;padding:6px 14px;border:1px solid var(--line2);background:#fff;
  border-radius:3px;cursor:pointer;color:var(--ink)}
button:hover{background:var(--panel)}
button.pri{border-color:var(--accent);color:var(--accent)}
input[type=range]{flex:1;min-width:180px;accent-color:var(--accent)}
.readout{font-variant-numeric:tabular-nums;font-size:13px;color:var(--muted);white-space:nowrap}
h2{font-size:13px;font-weight:600;text-transform:uppercase;letter-spacing:.08em;color:var(--faint);
  margin:0 0 10px}
.events{max-height:520px;overflow-y:auto;border:1px solid var(--line);border-radius:3px}
.ev{display:flex;gap:9px;align-items:flex-start;padding:8px 10px;border-bottom:1px solid var(--line);cursor:pointer}
.ev:last-child{border-bottom:none}
.ev:hover{background:var(--panel)}
.ev.now{background:#eef4ff}
.dot{width:8px;height:8px;border-radius:50%;flex:0 0 8px;margin-top:6px}
.ev .t{font-variant-numeric:tabular-nums;color:var(--faint);font-size:12px;flex:0 0 46px}
.ev .b{min-width:0}
.ev .k{font-size:13px;font-weight:500}
.ev .d{font-size:11.5px;color:var(--muted);word-break:break-word}
.scores{border:1px solid var(--line);border-radius:3px;overflow:hidden;margin-top:22px}
.sc{display:flex;justify-content:space-between;gap:10px;padding:7px 12px;border-bottom:1px solid var(--line);font-size:12.5px}
.sc:last-child{border-bottom:none}
.sc b{font-weight:600}
.grade{font-weight:600;padding:0 6px;border-radius:3px;font-size:11.5px}
.gA{background:#eaf3de;color:#27500A}.gB{background:#e1f5ee;color:#0f6e56}
.gC{background:#faeeda;color:#633806}.gD,.gE{background:#fcebeb;color:#8a2222}
.legend{display:flex;flex-wrap:wrap;gap:14px;margin-top:10px;font-size:12px;color:var(--muted)}
.legend span{display:flex;align-items:center;gap:5px}
.sw{width:10px;height:10px;border-radius:2px;display:inline-block}
.note{font-size:12px;color:var(--faint);margin-top:14px;line-height:1.7}
</style>
</head>
<body>
<div class="wrap">
  <h1>__TITLE__</h1>
  <div class="sub">交互式回放面板 · 示意定位图，非视频叠加</div>

  <div class="meta">
    <div>来源<b id="m-src">—</b></div>
    <div>帧率<b id="m-fps">—</b></div>
    <div>画面<b id="m-size">—</b></div>
    <div>跟踪器<b id="m-track">—</b></div>
    <div>标定<b id="m-calib">—</b></div>
    <div>事件<b id="m-events">—</b></div>
  </div>

  <div class="grid">
    <div>
      <canvas id="map" role="img" aria-label="带球员与球位置的球场示意图"></canvas>
      <div class="ctrl">
        <button id="play" class="pri">播放</button>
        <input type="range" id="scrub" min="0" max="100" value="0" aria-label="时间轴">
        <span class="readout" id="readout">—</span>
      </div>
      <div class="legend" id="legend"></div>
      <div class="note">
        位置来自跟踪输出，运动镜头下画面坐标会随镜头平移，因此本图为示意而非场地精确投影。
        拖动时间轴或点击右侧事件可跳转。
      </div>
    </div>
    <div>
      <h2>事件</h2>
      <div class="events" id="events"></div>
    </div>
  </div>

  <div class="scores" id="scores"></div>
</div>

<script>
const DATA = __DATA__;
const TEAM_KEYS = ["A","B","official","cluster3","cluster4","unknown"];
const TEAM_NAMES = {A:"A 队", B:"B 队", official:"裁判", cluster3:"第 3 组", cluster4:"第 4 组", unknown:"未分队"};
const COL = [DATA.teamColours.A, DATA.teamColours.B, DATA.teamColours.official,
             DATA.teamColours.cluster3, DATA.teamColours.cluster4, DATA.teamColours.unknown];

const byFrame = new Map();
for (const p of DATA.players) {
  if (!byFrame.has(p[0])) byFrame.set(p[0], []);
  byFrame.get(p[0]).push(p);
}
const ballAt = new Map();
for (const b of DATA.ball) ballAt.set(b[0], b);
const posAt = new Map();
for (const s of DATA.possession) {
  for (let f = s.start; f <= s.end; f++) posAt.set(f, s);
}

const first = DATA.meta.first_frame, last = DATA.meta.last_frame;
const W = DATA.meta.width, H = DATA.meta.height;
const canvas = document.getElementById("map");
const ctx = canvas.getContext("2d");
let frame = first, playing = false, raf = null;

function fit() {
  const dpr = window.devicePixelRatio || 1;
  canvas.width = Math.round(canvas.clientWidth * dpr);
  canvas.height = Math.round(canvas.clientWidth * H / W * dpr);
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  draw();
}
function sx(x){ return x / W * canvas.clientWidth; }
function sy(y){ return y / H * (canvas.clientWidth * H / W); }

function draw() {
  const w = canvas.clientWidth, h = canvas.clientWidth * H / W;
  ctx.clearRect(0, 0, w, h);
  ctx.fillStyle = "#fbfcfd"; ctx.fillRect(0, 0, w, h);

  ctx.strokeStyle = "#eceff3"; ctx.lineWidth = 1;
  for (let i = 1; i < 8; i++) {
    ctx.beginPath(); ctx.moveTo(w * i / 8, 0); ctx.lineTo(w * i / 8, h); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(0, h * i / 8); ctx.lineTo(w, h * i / 8); ctx.stroke();
  }

  ctx.strokeStyle = "rgba(47,111,237,.18)"; ctx.lineWidth = 1.4;
  ctx.beginPath();
  let started = false;
  for (const b of DATA.ball) {
    if (b[0] > frame) break;
    const px = sx(b[1]), py = sy(b[2]);
    if (!started) { ctx.moveTo(px, py); started = true; } else { ctx.lineTo(px, py); }
  }
  ctx.stroke();

  const poss = posAt.get(frame);
  if (poss) {
    ctx.fillStyle = "rgba(18,163,122,.07)";
    ctx.fillRect(0, 0, w, 20);
    ctx.fillStyle = "#0f6e56"; ctx.font = "12px system-ui,sans-serif";
    ctx.fillText("控球 #" + poss.track + " (" + (TEAM_NAMES[poss.team] || poss.team) + ")", 8, 14);
  }

  const near = DATA.events.filter(e => e.frame <= frame && e.frame > frame - 24);
  for (const e of near) {
    const b = ballAt.get(e.frame);
    let px, py;
    if (b) { px = sx(b[1]); py = sy(b[2]); }
    else {
      const ps = byFrame.get(e.frame);
      if (!ps || !ps.length) continue;
      const p = ps.find(q => q[1] === e.track) || ps[0];
      px = sx(p[2]); py = sy(p[3]);
    }
    ctx.beginPath(); ctx.arc(px, py, 13, 0, Math.PI * 2);
    ctx.strokeStyle = e.colour; ctx.lineWidth = 1.6; ctx.stroke();
  }

  const ps = byFrame.get(frame) || [];
  for (const p of ps) {
    ctx.beginPath(); ctx.arc(sx(p[2]), sy(p[3]), 5.5, 0, Math.PI * 2);
    ctx.fillStyle = COL[p[4]] || COL[5]; ctx.fill();
    ctx.strokeStyle = "#ffffff"; ctx.lineWidth = 1.2; ctx.stroke();
    ctx.fillStyle = "#4b5563"; ctx.font = "10px system-ui,sans-serif";
    ctx.fillText("#" + p[1], sx(p[2]) + 7, sy(p[3]) + 3);
  }

  const b = ballAt.get(frame);
  if (b) {
    ctx.beginPath(); ctx.arc(sx(b[1]), sy(b[2]), 6, 0, Math.PI * 2);
    ctx.fillStyle = b[3] ? "#b4b2a9" : "#111827"; ctx.fill();
    ctx.strokeStyle = "#fff"; ctx.lineWidth = 1.4; ctx.stroke();
  }

  document.getElementById("readout").textContent =
    "帧 " + frame + " / " + last + "   " + (frame / DATA.meta.fps).toFixed(2) + " s   " +
    ps.length + " 名球员";
  highlight();
}

function highlight() {
  let best = null, bestD = 1e9;
  document.querySelectorAll(".ev").forEach(el => {
    const f = +el.dataset.frame;
    const d = Math.abs(f - frame);
    if (f <= frame && d < bestD) { bestD = d; best = el; }
  });
  document.querySelectorAll(".ev").forEach(el => el.classList.remove("now"));
  if (best && bestD < 40) best.classList.add("now");
}

function setFrame(f) {
  frame = Math.max(first, Math.min(last, Math.round(f)));
  document.getElementById("scrub").value = frame;
  draw();
}

document.getElementById("scrub").min = first;
document.getElementById("scrub").max = last;

document.getElementById("play").addEventListener("click", () => {
  playing = !playing;
  document.getElementById("play").textContent = playing ? "暂停" : "播放";
  if (playing) step();
});
function step() {
  if (!playing) return;
  setFrame(frame + Math.max(1, Math.round(DATA.meta.fps / 8)));
  if (frame >= last) { playing = false; document.getElementById("play").textContent = "播放"; return; }
  raf = requestAnimationFrame(step);
}
document.getElementById("scrub").addEventListener("input", e => {
  playing = false; document.getElementById("play").textContent = "播放";
  setFrame(+e.target.value);
});
canvas.addEventListener("click", e => {
  const r = canvas.getBoundingClientRect();
  const x = (e.clientX - r.left) / r.width * W, y = (e.clientY - r.top) / r.height * H;
  let best = null, bestD = 1e9;
  for (const p of (byFrame.get(frame) || [])) {
    const d = (p[2] - x) ** 2 + (p[3] - y) ** 2;
    if (d < bestD) { bestD = d; best = p; }
  }
  if (best && bestD < 60 * 60 && best[1] != null) {
    document.querySelectorAll(".ev").forEach(el => {
      if (+el.dataset.track === best[1]) el.style.background = "#eef4ff";
      else el.style.background = "";
    });
  }
});

const evBox = document.getElementById("events");
for (const e of DATA.events) {
  const el = document.createElement("div");
  el.className = "ev"; el.dataset.frame = e.frame; el.dataset.track = e.track;
  el.innerHTML =
    '<span class="dot" style="background:' + e.colour + '"></span>' +
    '<span class="t">' + e.t.toFixed(2) + 's</span>' +
    '<span class="b"><span class="k">' + e.label + (e.track != null ? ' · #' + e.track : '') + '</span>' +
    '<div class="d">' + e.detail + '</div></span>';
  el.addEventListener("click", () => {
    playing = false; document.getElementById("play").textContent = "播放";
    setFrame(Math.max(first, e.frame - 8));
  });
  evBox.appendChild(el);
}

const sc = DATA.scorecard;
const scBox = document.getElementById("scores");
scBox.innerHTML = '<div class="sc"><span>输出质量自检（无标注）</span><span><b>' +
  sc.overall + '/100</b> <span class="grade g' + sc.grade + '">' + sc.grade + '</span></span></div>' +
  Object.entries(sc.metrics).map(([k, m]) =>
    '<div class="sc"><span>' + k + ' <span style="color:var(--faint)">' +
    m.value.toFixed(3) + ' ' + m.unit + '</span></span><span><span class="grade g' + m.grade + '">' +
    m.grade + '</span></span></div>').join('');

const lg = document.getElementById("legend");
lg.innerHTML = TEAM_KEYS.map((k, i) => ({ k, i }))
  .filter(o => DATA.teamLabels.indexOf(o.i) >= 0)
  .map(o => '<span><i class="sw" style="background:' + COL[o.i] + '"></i>' + TEAM_NAMES[o.k] + '</span>').join('') +
  Object.entries(DATA.eventLabels).map(([k, v]) =>
    '<span><i class="sw" style="background:' + DATA.eventColours[k] + '"></i>' + v + ' 事件</span>').join('');

document.getElementById("m-src").textContent = DATA.meta.source;
document.getElementById("m-fps").textContent = DATA.meta.fps + " fps";
document.getElementById("m-size").textContent = DATA.meta.width + "x" + DATA.meta.height;
document.getElementById("m-track").textContent = DATA.meta.tracker;
document.getElementById("m-calib").textContent = DATA.meta.calibration;
document.getElementById("m-events").textContent = DATA.events.length + " 条";

window.addEventListener("resize", fit);
fit();
</script>
</body>
</html>
"""


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if not os.path.exists(args.trajectory):
        print(f"[error] not found: {args.trajectory}", file=sys.stderr)
        return 2
    out_dir = args.out or os.path.dirname(os.path.abspath(args.trajectory))
    os.makedirs(out_dir, exist_ok=True)

    payload = collect(args.trajectory, args.video, args.no_teams, args.teams)
    title = args.title or f"比赛回放面板 · {payload['meta']['source']}"
    html = HTML.replace("__DATA__", json.dumps(payload, ensure_ascii=False)).replace("__TITLE__", title)

    path = os.path.join(out_dir, "dashboard.html")
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(html)
    size_kb = os.path.getsize(path) / 1024
    print(f"[ok] wrote {path} ({size_kb:.0f} KB, self-contained)")
    print(f"     {len(payload['players'])} player rows, {len(payload['ball'])} ball points, "
          f"{len(payload['events'])} events, {len(payload['possession'])} possession segments")
    if payload["meta"]["calibration"] == "none":
        print("     note: clip is uncalibrated, so the map is schematic")
    return 0


if __name__ == "__main__":
    sys.exit(main())
