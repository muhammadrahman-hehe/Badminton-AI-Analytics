import cv2, json, os, numpy as np, pandas as pd

# ---- settings ----
NAMES       = {"P1": "ANTONSEN", "P2": "NG K.L.A."}
START_SCORE = {"P1": 1, "P2": 1}
COL         = {"P1": (60, 60, 230), "P2": (210, 190, 60)}     # BGR: red shirt / light-blue shirt
TITLE       = "BADMINTON AI ANALYTICS"
OUT         = "outputs/dashboard.mp4"

W, L = 6.1, 13.4
SINGLES = (0.46, W - 0.46)

# ---- data ----
pts = np.array(json.load(open("court.json")), np.float32)
cap = cv2.VideoCapture("clip_new.mp4")
w, h, fps, n = int(cap.get(3)), int(cap.get(4)), cap.get(5), int(cap.get(7))
k = h / 1080                                   # scale everything for other resolutions
PW = int(700 * k)                              # side panel width

b = pd.read_csv("outputs/clip_ball.csv").set_index("Frame").reindex(range(n))
vis = b.Visibility == 1
X = b.X.where(vis).astype(float).values
Y = b.Y.where(vis).astype(float).values

hits = pd.read_csv("outputs/hits_rally.csv" if os.path.exists("outputs/hits_rally.csv") else "outputs/hits.csv")
hits = hits.sort_values("frame")
ral = pd.read_csv("outputs/rallies.csv").sort_values("rally")
pl = pd.read_csv("outputs/players_clean.csv")
box = {p: g.set_index("frame").reindex(range(n)) for p, g in pl.groupby("player")}

# distance covered (metres): sample every 5 frames, ignore tracking jumps
dist = {}
for p, g in box.items():
    cx, cy = g.cx.values, g.cy.values
    d = np.zeros(n)
    for f in range(5, n):
        step = 0.0
        if f % 5 == 0 and not (np.isnan(cx[f]) or np.isnan(cx[f-5])):
            s = np.hypot(cx[f] - cx[f-5], cy[f] - cy[f-5])
            step = s if s < 2.0 else 0.0
        d[f] = d[f-1] + step
    dist[p] = d

# ---- drawing helpers ----
FONT = cv2.FONT_HERSHEY_SIMPLEX
def text(img, s, x, y, scale=0.8, color=(255,255,255), th=2):
    cv2.putText(img, s, (int(x), int(y)), FONT, scale * k, color, max(1, int(th * k)), cv2.LINE_AA)

# court drawing in the panel
CH = int(600 * k)                              # court area height
CPAD = int(18 * k)
S = (CH - 2 * CPAD) / L                        # px per metre
CW = int(W * S + 2 * CPAD)
def P(x, y): return (int(x * S) + CPAD, int(y * S) + CPAD)

court_base = np.full((CH, CW, 3), (45, 110, 45), np.uint8)
wc = (235, 235, 235)
cv2.rectangle(court_base, P(0, 0), P(W, L), wc, 1)
for x in SINGLES: cv2.line(court_base, P(x, 0), P(x, L), wc, 1)
for y in (0.76, L - 0.76, L/2 - 1.98, L/2 + 1.98): cv2.line(court_base, P(0, y), P(W, y), wc, 1)
cv2.line(court_base, P(W/2, 0), P(W/2, L/2 - 1.98), wc, 1)
cv2.line(court_base, P(W/2, L/2 + 1.98), P(W/2, L), wc, 1)
cv2.line(court_base, P(0, L/2), P(W, L/2), (255, 255, 255), 2)

heat = np.zeros((CH, CW), np.float32)

# ---- per-frame lookups ----
hit_at = {int(x.frame): x for x in hits.itertuples()}
rally_rows = list(ral.itertuples())
def rally_state(i):
    """current rally (or the last finished one) at frame i"""
    cur = None
    for r in rally_rows:
        if r.start_frame <= i: cur = r
    return cur

# ---- render ----
out = cv2.VideoWriter(OUT, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w + PW, h))
score = dict(START_SCORE)
shots_total = {"P1": 0, "P2": 0}
trail_s, trail_p = [], {"P1": [], "P2": []}
hit_marks, land_marks, results = [], [], []
last_hit, banner = None, None
i = 0
while True:
    ok, f = cap.read()
    if not ok: break

    # events
    if i in hit_at:
        hh = hit_at[i]; last_hit = (hh, i); shots_total[hh.player] += 1
        hit_marks.append((hh.player, hh.cx, hh.cy))
    for r in rally_rows:
        if r.end_frame == i:
            score = {"P1": int(r.score_P1), "P2": int(r.score_P2)}
            est = " est." if "estimated" in str(r.landing) else ""
            res = str(r.result).split(" (")[0]
            banner = (f"POINT {r.winner}  ({res}{est})", i)
            if not np.isnan(r.landing_cx): land_marks.append((r.rally, r.landing_cx, r.landing_cy))
            results.append(f"R{r.rally}  {r.winner:<10} {res:<4} {int(r.shots)} shots {r.duration_s:.1f}s{est}")

    # ---------- video overlays ----------
    p = None if np.isnan(X[i]) else (int(X[i]), int(Y[i]))
    trail_s = (trail_s + [p])[-10:]
    for j, q in enumerate(trail_s):
        if q: cv2.circle(f, q, max(2, int((2 + j * 0.5) * k)), (0, 255, 255), -1)
    for pname, g in box.items():
        r = g.loc[i]
        if not np.isnan(r.x1):
            cv2.rectangle(f, (int(r.x1), int(r.y1)), (int(r.x2), int(r.y2)), COL[pname], max(2, int(3 * k)))
            text(f, NAMES[pname], r.x1, r.y1 - 10 * k, 0.8, COL[pname], 2)
            trail_p[pname] = (trail_p[pname] + [P(r.cx, r.cy)])[-40:]
            heat[np.clip(P(r.cx, r.cy)[1], 0, CH-1), np.clip(P(r.cx, r.cy)[0], 0, CW-1)] += 1
    if last_hit and i - last_hit[1] < 18:
        hh = last_hit[0]
        cv2.circle(f, (int(hh.x), int(hh.y)), int(32 * k), (0, 255, 255), max(2, int(3 * k)))
        text(f, ("SERVE " if hh.serve else "HIT ") + NAMES[hh.player], hh.x + 38 * k, hh.y, 0.9, (0, 255, 255), 2)
    if banner and i - banner[1] < 60:
        (tw, th), _ = cv2.getTextSize(banner[0], FONT, 1.5 * k, int(3 * k))
        x0 = (w - tw) // 2
        cv2.rectangle(f, (x0 - 25, int(140 * k)), (x0 + tw + 25, int(140 * k) + th + int(34 * k)), (0, 0, 0), -1)
        text(f, banner[0], x0, 140 * k + th + 16 * k, 1.5, (0, 255, 255), 3)

    # ---------- side panel ----------
    pan = np.full((h, PW, 3), (28, 24, 22), np.uint8)
    y = 55 * k
    text(pan, TITLE, 25 * k, y, 0.95, (0, 220, 255), 2)
    cv2.line(pan, (int(25 * k), int(75 * k)), (PW - int(25 * k), int(75 * k)), (80, 80, 80), 1)

    # scoreboard
    cur = rally_state(i)
    nxt = [r for r in rally_rows if r.start_frame > i]
    serving = None
    if cur is not None and i <= cur.end_frame: serving = cur.server
    elif nxt: serving = nxt[0].server
    for row, pn in enumerate(("P1", "P2")):
        yy = int((120 + row * 70) * k)
        cv2.rectangle(pan, (int(25 * k), yy - int(45 * k)), (PW - int(25 * k), yy + int(15 * k)), (50, 44, 40), -1)
        cv2.rectangle(pan, (int(25 * k), yy - int(45 * k)), (int(35 * k), yy + int(15 * k)), COL[pn], -1)
        text(pan, NAMES[pn], 50 * k, yy, 1.0, (255, 255, 255), 2)
        if serving == NAMES[pn]: cv2.circle(pan, (int(PW - 150 * k), yy - int(15 * k)), int(9 * k), (0, 255, 255), -1)
        text(pan, str(score[pn]), PW - 95 * k, yy + 2 * k, 1.5, (255, 255, 255), 3)

    # rally info
    y = 255 * k
    if cur is not None:
        live = i <= cur.end_frame
        t = (min(i, cur.end_frame) - cur.start_frame) / fps
        n_shots = int(((hits.rally == cur.rally) & (hits.frame <= i)).sum())
        status = "LIVE" if live else "ENDED"
        text(pan, f"Rally {cur.rally}  [{status}]", 25 * k, y, 0.85, (0, 255, 0) if live else (160, 160, 160), 2)
        text(pan, f"Shots {n_shots}    Time {t:4.1f}s", 25 * k, y + 38 * k, 0.85, (255, 255, 255), 2)
    else:
        text(pan, "Waiting for serve...", 25 * k, y, 0.85, (160, 160, 160), 2)

    # shots & distance bars
    y = 345 * k
    text(pan, "SHOTS", 25 * k, y, 0.7, (180, 180, 180), 2)
    text(pan, "DISTANCE (m)", PW / 2 + 10 * k, y, 0.7, (180, 180, 180), 2)
    mx_s = max(1, max(shots_total.values()))
    mx_d = max(1.0, max(dist[p][i] for p in dist))
    for row, pn in enumerate(("P1", "P2")):
        yy = int(y + (30 + row * 34) * k)
        bw = int((PW / 2 - 110 * k) * shots_total[pn] / mx_s)
        cv2.rectangle(pan, (int(25 * k), yy - int(18 * k)), (int(25 * k) + bw, yy + int(4 * k)), COL[pn], -1)
        text(pan, str(shots_total[pn]), 25 * k + bw + 8 * k, yy, 0.7, (255, 255, 255), 2)
        dw = int((PW / 2 - 110 * k) * dist[pn][i] / mx_d)
        x0 = PW / 2 + 10 * k
        cv2.rectangle(pan, (int(x0), yy - int(18 * k)), (int(x0) + dw, yy + int(4 * k)), COL[pn], -1)
        text(pan, f"{dist[pn][i]:.0f}", x0 + dw + 8 * k, yy, 0.7, (255, 255, 255), 2)

    # court: heatmap + trails + hits + landings
    court = court_base.copy()
    if heat.max() > 0:
        hm = cv2.GaussianBlur(heat, (0, 0), 10 * k)
        hm = (255 * hm / hm.max()).astype(np.uint8)
        cm = cv2.applyColorMap(hm, cv2.COLORMAP_JET)
        mask = hm > 25
        court[mask] = (0.55 * court[mask] + 0.45 * cm[mask]).astype(np.uint8)
        cv2.rectangle(court, P(0, 0), P(W, L), wc, 1)
        cv2.line(court, P(0, L/2), P(W, L/2), (255, 255, 255), 2)
    for pn, hx, hy in hit_marks[-30:]:
        cv2.drawMarker(court, P(hx, hy), COL[pn], cv2.MARKER_CROSS, int(10 * k), 2)
    for rid, lx, ly in land_marks:
        cv2.drawMarker(court, P(lx, ly), (0, 255, 255), cv2.MARKER_TILTED_CROSS, int(16 * k), 3)
        text(court, str(rid), P(lx, ly)[0] + 8 * k, P(lx, ly)[1] - 6 * k, 0.5, (0, 255, 255), 1)
    for pn, tr in trail_p.items():
        for a, c in zip(tr, tr[1:]): cv2.line(court, a, c, COL[pn], 2)
        if tr and not np.isnan(box[pn].loc[i].cx): cv2.circle(court, tr[-1], int(8 * k), COL[pn], -1)

    cy0 = int(468 * k)
    text(pan, "COURT  heatmap / hits / landings", 25 * k, cy0 - 12 * k, 0.6, (180, 180, 180), 1)
    pan[cy0:cy0 + CH, int(25 * k):int(25 * k) + CW] = court

    # legend + rally results next to the court
    lx = int(25 * k) + CW + int(20 * k)
    ly = cy0 + int(25 * k)
    for pn in ("P1", "P2"):
        cv2.circle(pan, (lx + int(8 * k), ly - int(6 * k)), int(7 * k), COL[pn], -1)
        text(pan, NAMES[pn], lx + 22 * k, ly, 0.6, (230, 230, 230), 1); ly += int(28 * k)
    cv2.drawMarker(pan, (lx + int(8 * k), ly - int(6 * k)), (220, 220, 220), cv2.MARKER_CROSS, int(12 * k), 2)
    text(pan, "hit position", lx + 22 * k, ly, 0.6, (230, 230, 230), 1); ly += int(28 * k)
    cv2.drawMarker(pan, (lx + int(8 * k), ly - int(6 * k)), (0, 255, 255), cv2.MARKER_TILTED_CROSS, int(12 * k), 2)
    text(pan, "landing", lx + 22 * k, ly, 0.6, (230, 230, 230), 1); ly += int(50 * k)
    text(pan, "RALLY RESULTS", lx, ly, 0.65, (0, 220, 255), 2); ly += int(30 * k)
    for line in results[-8:]:
        parts = line.split("  ", 1)
        text(pan, parts[0], lx, ly, 0.55, (255, 255, 255), 1)
        text(pan, parts[1].strip(), lx, ly + 22 * k, 0.5, (200, 200, 200), 1)
        ly += int(52 * k)

    out.write(np.hstack([f, pan])); i += 1
    if i % 100 == 0: print(i, "frames")
out.release()
print("done", i, "frames ->", OUT, f"({w + PW}x{h})")
