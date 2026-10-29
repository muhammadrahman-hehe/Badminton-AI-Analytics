import cv2, json, numpy as np, pandas as pd

# ---- settings ----
NAMES       = {"P1": "ANTONSEN", "P2": "NG K.L.A."}
START_SCORE = {"P1": 1, "P2": 1}  # points on the scoreboard when the clip's first rally is served
STILL_SPEED  = 3    # px/frame: shuttle lying on the floor
STILL_FRAMES = 6    # frames it must lie still to count as landed
LOST_FRAMES  = 15   # shuttle not visible this long after the last hit -> use last seen point
MAX_SEARCH   = 150  # frames to look for the landing after the last hit
TOL          = 0.05 # metres: shuttle on the line = IN

W, L, PAD = 6.1, 13.4, 20
SINGLES = (0.46, W - 0.46)          # singles side lines
other = {"P1": "P2", "P2": "P1"}

pts = np.array(json.load(open("court.json")), np.float32)
H = cv2.getPerspectiveTransform(pts, np.array([[0,0],[W,0],[W,L],[0,L]], np.float32))
def to_court(x, y):
    c = cv2.perspectiveTransform(np.array([[[x, y]]], np.float32), H)[0, 0]
    return float(c[0]), float(c[1])

cap = cv2.VideoCapture("clip_new.mp4")
w, h, fps, n = int(cap.get(3)), int(cap.get(4)), cap.get(5), int(cap.get(7))

b = pd.read_csv("outputs/clip_ball.csv").set_index("Frame").reindex(range(n))
vis = b.Visibility == 1
Xr = b.X.where(vis).astype(float).values          # raw shuttle positions (no gap filling)
Yr = b.Y.where(vis).astype(float).values

hits = pd.read_csv("outputs/hits.csv").sort_values("frame")
pl = pd.read_csv("outputs/players_clean.csv")
box = {k: g.set_index("frame") for k, g in pl.groupby("player")}

# ---- landing: first moment the shuttle lies still after the last hit ----
def find_landing(f0, stop, still_frames=STILL_FRAMES):
    end = min(stop, f0 + MAX_SEARCH, n)
    run, prev, last_seen, lost = 0, None, None, 0
    for f in range(f0 + 2, end):
        if np.isnan(Xr[f]):
            lost += 1
            if lost >= LOST_FRAMES and last_seen is not None:
                return last_seen, "last seen"
            continue
        lost = 0
        if prev is not None and np.hypot(Xr[f] - Xr[prev], Yr[f] - Yr[prev]) < STILL_SPEED * (f - prev):
            run += 1
            if run >= still_frames:
                return f - run, "landed"
        else:
            run = 0
        prev = last_seen = f
    if end == n and last_seen is not None and last_seen > n - 10:
        return last_seen, "clip end (estimated)"   # video stops before the shuttle settles
    return None, "not found"

serve_rows = hits[hits.serve.astype(bool)].sort_values("frame")
rallies = sorted(hits.rally.unique())
rows, score, dropped = [], dict(START_SCORE), []
for k, r_id in enumerate(rallies):
    g = hits[hits.rally == r_id].sort_values("frame")
    # a rally ends as soon as the shuttle lands; "hits" after that are taps between points
    fr = g.frame.astype(int).tolist()
    for j in range(len(fr) - 1):
        if fr[j + 1] - fr[j] < 40: continue            # normal shot-to-shot time, rally goes on
        _, how_j = find_landing(fr[j], fr[j + 1], still_frames=10)
        if how_j == "landed":
            print(f"  rally {r_id}: shuttle landed after frame {fr[j]} -> ignoring later taps {fr[j+1:]}")
            dropped += fr[j + 1:]
            g = g.iloc[:j + 1]
            break
    first, last = g.iloc[0], g.iloc[-1]
    server = first.player
    nxt = serve_rows[serve_rows.frame > last.frame]
    next_server = nxt.iloc[0].player if len(nxt) else None
    stop = int(nxt.iloc[0].frame) if len(nxt) else n

    lf, how = find_landing(int(last.frame), stop)
    hitter, opp = last.player, other[last.player]
    lcx = lcy = np.nan
    result, winner = "unknown", None
    if lf is not None:
        lcx, lcy = to_court(Xr[lf], Yr[lf])
        in_opp_half = (lcy < L / 2) if opp == "P1" else (lcy >= L / 2)   # P1 = far half
        inside = SINGLES[0] - TOL <= lcx <= SINGLES[1] + TOL and -TOL <= lcy <= L + TOL
        result = "net" if not in_opp_half else ("in" if inside else "out")
        winner = hitter if result == "in" else opp

    # badminton rule: the winner of a rally serves the next one -> use it to confirm / fill in
    if next_server is not None:
        if winner is None:
            winner, result = next_server, result + " (from next serve)"
        elif winner != next_server:
            print(f"  rally {r_id}: landing says {NAMES[winner]}, next serve says {NAMES[next_server]} -> using next serve")
            winner, result = next_server, result + " (corrected by next serve)"

    end_frame = lf if lf is not None else int(last.frame) + 30
    if winner: score[winner] += 1
    rows.append(dict(rally=int(r_id), server=NAMES[server], start_frame=int(first.frame),
                     end_frame=int(end_frame), duration_s=round((end_frame - first.frame) / fps, 1),
                     shots=len(g), shots_P1=int((g.player == "P1").sum()), shots_P2=int((g.player == "P2").sum()),
                     last_hitter=NAMES[hitter], landing=how, landing_cx=round(lcx, 2), landing_cy=round(lcy, 2),
                     result=result, winner=NAMES[winner] if winner else "?",
                     score_P1=score["P1"], score_P2=score["P2"]))

rd = pd.DataFrame(rows)
rd.to_csv("outputs/rallies.csv", index=False)
pd.set_option("display.width", 220)
print(rd.to_string(index=False))
print(f"final score in clip: {NAMES['P1']} {score['P1']} - {score['P2']} {NAMES['P2']}")

# ---- minimap ----
S = int((h * 0.6 - 2 * PAD) / L)
def P(x, y): return (int(x * S) + PAD, int(y * S) + PAD)
base = np.full((int(L*S) + 2*PAD, int(W*S) + 2*PAD, 3), (60,120,60), np.uint8)
white = (255,255,255)
cv2.rectangle(base, P(0,0), P(W,L), white, 1)
for x in SINGLES: cv2.line(base, P(x,0), P(x,L), white, 1)
for y in (0.76, L-0.76, L/2-1.98, L/2+1.98): cv2.line(base, P(0,y), P(W,y), white, 1)
cv2.line(base, P(W/2,0), P(W/2,L/2-1.98), white, 1)
cv2.line(base, P(W/2,L/2+1.98), P(W/2,L), white, 1)
cv2.line(base, P(0,L/2), P(W,L/2), (200,200,200), 2)

# ---- check video ----
col = {"P1": (255,128,0), "P2": (0,0,255)}
hits[~hits.frame.isin(dropped)].to_csv("outputs/hits_rally.csv", index=False)
hit_at = {int(x.frame): x for x in hits.itertuples() if int(x.frame) not in dropped}
end_at = {r["end_frame"]: r for r in rows}
shown = dict(START_SCORE)
cur_rally, shots, banner, landed, last_hit, trail = 0, 0, None, [], None, []
out = cv2.VideoWriter("outputs/rally_check.mp4", cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
i = 0
while True:
    ok, f = cap.read()
    if not ok: break
    if i in hit_at:
        hh = hit_at[i]
        if hh.rally != cur_rally: cur_rally, shots = hh.rally, 0
        shots += 1; last_hit = (hh, i)
    if i in end_at:
        r = end_at[i]
        shown = {"P1": r["score_P1"], "P2": r["score_P2"]}
        est = "  est." if "estimated" in r["landing"] else ""
        banner = (f"POINT {r['winner']}  ({r['result'].split(' (')[0]}{est})", i)
        if not np.isnan(r["landing_cx"]): landed.append((P(r["landing_cx"], r["landing_cy"]), r["rally"]))

    p = None if np.isnan(Xr[i]) else (int(Xr[i]), int(Yr[i]))
    trail = (trail + [p])[-10:]
    for q in trail:
        if q: cv2.circle(f, q, 5, (0,255,255), -1)

    mm = base.copy()
    for name, g in box.items():
        if i in g.index:
            r = g.loc[i]
            cv2.rectangle(f, (int(r.x1), int(r.y1)), (int(r.x2), int(r.y2)), col[name], 2)
            cv2.circle(mm, P(r.cx, r.cy), 6, col[name], -1)
    for m, rid in landed:
        cv2.drawMarker(mm, m, (0,255,255), cv2.MARKER_TILTED_CROSS, 14, 3)
        cv2.putText(mm, str(rid), (m[0] + 8, m[1] - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 1)

    if last_hit and i - last_hit[1] < 20:
        hh = last_hit[0]
        label = ("SERVE " if hh.serve else "HIT ") + NAMES[hh.player]
        cv2.circle(f, (int(hh.x), int(hh.y)), 30, (0,255,255), 3)
        cv2.putText(f, label, (int(hh.x) + 35, int(hh.y)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,255), 2)

    # score panel
    cv2.rectangle(f, (30, 120), (640, 235), (0,0,0), -1)
    cv2.putText(f, f"{NAMES['P1']} {shown['P1']} - {shown['P2']} {NAMES['P2']}", (40, 160),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255), 2)
    cv2.putText(f, f"Rally {cur_rally}   Shots {shots}   frame {i}", (40, 215),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

    if banner and i - banner[1] < 60:
        (tw, th), _ = cv2.getTextSize(banner[0], cv2.FONT_HERSHEY_SIMPLEX, 1.4, 3)
        x0 = (w - tw) // 2
        cv2.rectangle(f, (x0 - 20, 250), (x0 + tw + 20, 250 + th + 30), (0,0,0), -1)
        cv2.putText(f, banner[0], (x0, 250 + th + 12), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0,255,255), 3)

    mh, mw = mm.shape[:2]
    f[10:10+mh, w-mw-10:w-10] = mm
    out.write(f); i += 1
out.release()
print("done", i, "frames -> outputs/rally_check.mp4")