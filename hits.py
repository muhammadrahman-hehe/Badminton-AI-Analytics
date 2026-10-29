import cv2, json, numpy as np, pandas as pd

# ---- tuning knobs ----
K           = 4     # frames before/after used to measure shuttle velocity
DV_MIN      = 50    # min sudden velocity change (pixels) to count as a hit (fakes were ~40)
NEAR_PX     = 130   # shuttle must be this close (px) to a player's box
GAP         = 12    # min frames between two hits
MIN_TRAVEL  = 80    # shuttle must move at least this far (px) in the next 12 frames
RALLY_GAP   = 75    # frames without a hit (2.5 s) = rally over
HOLD_SPEED  = 8     # px/frame: below this the shuttle is "held" in the server's hand
HOLD_MIN    = 5     # frames the shuttle must be held before the serve
SERVE_DISP  = 50    # px the shuttle must move away within 15 frames after the serve
SERVE_SPEED  = 30   # fallback: fast launch (high serve) ...
SERVE_TRAVEL = 180  # ... that goes far
STILL_M, STILL_WIN, STILL_RUN, SERVE_AFTER = 0.25, 10, 15, 45

W, L, PAD = 6.1, 13.4, 20
pts = np.array(json.load(open("court.json")), np.float32)

cap = cv2.VideoCapture("clip_new.mp4")
w, h, fps, n = int(cap.get(3)), int(cap.get(4)), cap.get(5), int(cap.get(7))

# ---- shuttle (bridge gaps up to 10 frames: shuttle hides behind racket at contact) ----
b = pd.read_csv("outputs/clip_ball.csv").set_index("Frame").reindex(range(n))
vis = b.Visibility == 1
xs = b.X.where(vis).astype(float).interpolate(limit=10, limit_area="inside")
ys = b.Y.where(vis).astype(float).interpolate(limit=10, limit_area="inside")
X, Y = xs.values, ys.values

vbx, vby = xs - xs.shift(K), ys - ys.shift(K)
vax, vay = xs.shift(-K) - xs, ys.shift(-K) - ys
dv_s = np.hypot(vax - vbx, vay - vby)
DV = dv_s.values
peak = (dv_s.eq(dv_s.rolling(2*K+1, center=True, min_periods=1).max()) & (dv_s > DV_MIN)).values
SP = np.append(np.hypot(np.diff(X), np.diff(Y)), np.nan)      # shuttle speed px/frame

# ---- players ----
pl = pd.read_csv("outputs/players_clean.csv")
box = {k: g.set_index("frame") for k, g in pl.groupby("player")}
other = {"P1": "P2", "P2": "P1"}

def box_dist(name, f):
    g = box[name]
    if f not in g.index or np.isnan(X[f]): return 1e9
    r = g.loc[f]
    return float(np.hypot(max(r.x1 - X[f], 0, X[f] - r.x2), max(r.y1 - Y[f], 0, Y[f] - r.y2)))

def travel(f):
    seg = [(X[j], Y[j]) for j in range(f + 1, min(f + 13, n)) if not np.isnan(X[j])]
    return max((np.hypot(x - X[f], y - Y[f]) for x, y in seg), default=0)

def make_hit(f, who, serve=False):
    r = box[who].loc[f]
    return dict(frame=f, time_s=round(f / fps, 2), player=who, x=X[f], y=Y[f],
                dist=round(box_dist(who, f), 1), dv=round(float(np.nan_to_num(DV[f])), 1),
                cx=r.cx, cy=r.cy, serve=serve)

# ---- pass 1: candidate hits ----
hits = []
for f in np.where(peak)[0]:
    f = int(f)
    d = {p: box_dist(p, f) for p in ("P1", "P2")}
    who = min(d, key=d.get)
    if d[who] > NEAR_PX: continue
    # a strong hit may not travel far (e.g. the last shot of a rally goes into the net)
    if travel(f) < MIN_TRAVEL and DV[f] < 100: continue
    hit = make_hit(f, who)
    if hits and f - hits[-1]["frame"] < GAP:
        if hit["dv"] > hits[-1]["dv"]: hits[-1] = hit
        continue
    hits.append(hit)

# ---- serve set-ups: both players stand still ----
still = np.ones(n, bool)
for name in ("P1", "P2"):
    g = box[name].reindex(range(n))
    cx, cy = g.cx.values, g.cy.values
    d = np.full(n, np.inf)
    d[STILL_WIN:] = np.hypot(cx[STILL_WIN:] - cx[:-STILL_WIN], cy[STILL_WIN:] - cy[:-STILL_WIN])
    still &= np.nan_to_num(d, nan=np.inf) < STILL_M
runs, s = [], None
for f in range(n + 1):
    if f < n and still[f]:
        if s is None: s = f
    elif s is not None:
        if f - s >= STILL_RUN: runs.append((s, f - 1))
        s = None
print("serve set-ups (both players still):", runs)

# ---- serve = first moment the shuttle is LAUNCHED fast after a set-up ----
serves = []
VBY, VAY = vby.values, vay.values
def find_serve(s0, e):
    # 1) the server HOLDS the shuttle still, then serves: the serve is the moment the
    #    shuttle starts moving after being held (works for slow low/short serves too)
    end = min(e + SERVE_AFTER, n - 16)
    hold = 0
    for f in range(max(0, s0 - 10), end):
        if np.isnan(X[f]) or np.isnan(SP[f]): hold = 0; continue
        if SP[f] < HOLD_SPEED: hold += 1; continue
        if hold >= HOLD_MIN:
            ds = [np.hypot(X[j] - X[f], Y[j] - Y[f]) for j in range(f + 1, f + 16) if not np.isnan(X[j])]
            if ds and max(ds) >= SERVE_DISP: return f
        hold = 0
    # 2) fallback: first fast launch that goes far (high serve)
    end = min(e + SERVE_AFTER, n - K)
    for f in range(s0, end):
        if np.isnan(X[f]): continue
        if np.nanmax(SP[f:f+3]) >= SERVE_SPEED and travel(f) >= SERVE_TRAVEL:
            win = [j for j in range(max(s0, f - 3), min(f + 4, n)) if not np.isnan(DV[j])]
            return max(win, key=lambda j: DV[j]) if win else f   # exact contact = sharpest change
    return None

for s0, e in runs:
    f = find_serve(s0, e)
    if f is not None:
        # server = the player who held / bounced the shuttle during the set-up
        med = {}
        for p in ("P1", "P2"):
            ds = [box_dist(p, j) for j in range(s0, f + 1)]
            ds = [x for x in ds if x < 1e8]
            med[p] = float(np.median(ds)) if ds else 1e9
        who = min(med, key=med.get)
        if box_dist(who, f) > NEAR_PX * 1.5: continue
        serves.append(make_hit(f, who, serve=True))
        # drop pre-serve junk (bounces, holding) and duplicates of the serve itself
        hits = [x for x in hits if not (s0 <= x["frame"] < f + GAP)]
print("serves:", [(x["frame"], x["player"]) for x in serves])
hits = sorted(hits + serves, key=lambda x: x["frame"])

# ---- pass 2: enforce P1/P2 alternation (a serve is never replaced) ----
MISS_DV = 25   # a hit we KNOW must be there (same player twice in a row) may be weak

def reach_dist(name, f):
    # distance to the box grown by racket reach: up (overhead / high backhand) and sideways
    g = box[name]
    if f not in g.index or np.isnan(X[f]): return 1e9
    r = g.loc[f]
    bw, bh = r.x2 - r.x1, r.y2 - r.y1
    x1, x2, y1, y2 = r.x1 - 0.3 * bw, r.x2 + 0.3 * bw, r.y1 - 0.5 * bh, r.y2
    return float(np.hypot(max(x1 - X[f], 0, X[f] - x2), max(y1 - Y[f], 0, Y[f] - y2)))

def missed_between(a, b, who):
    best = None
    for f in range(a + GAP, b - GAP + 1):
        if np.isnan(DV[f]) or DV[f] < MISS_DV: continue
        if reach_dist(who, f) > NEAR_PX or travel(f) < MIN_TRAVEL: continue
        if best is None or DV[f] > DV[best]: best = f
    return best

fixed = []
for hit in hits:
    prev = fixed[-1] if fixed else None
    if hit["serve"] or not prev:
        fixed.append(hit); continue
    if hit["player"] == prev["player"] and hit["frame"] - prev["frame"] < RALLY_GAP:
        # 1) first look for the other player's (soft) hit in between, e.g. a net touch
        m = missed_between(prev["frame"], hit["frame"], other[hit["player"]])
        if m is not None:
            fixed.append(make_hit(m, other[hit["player"]])); fixed.append(hit)
        # 2) boxes overlap in the image (far player stands "behind" near player):
        #    if the other player is also in reach, it was really the other player's hit
        elif box_dist(other[hit["player"]], hit["frame"]) <= NEAR_PX:
            fixed.append(make_hit(hit["frame"], other[hit["player"]]))
        elif prev["serve"]:
            pass                                   # keep the serve, drop this one
        elif min(hit["dv"], prev["dv"]) > 150 and hit["frame"] - prev["frame"] > 30:
            fixed.append(hit)                      # both clearly real
        elif hit["dv"] > prev["dv"]:
            fixed[-1] = hit
        continue
    fixed.append(hit)
hits = fixed

# ---- rallies: start at a serve, end after RALLY_GAP without a hit ----
rally, in_rally, kept = 0, False, []
for hit in hits:
    if in_rally and hit["frame"] - kept[-1]["frame"] > RALLY_GAP: in_rally = False
    if hit["serve"]:
        rally += 1; in_rally = True
    if not in_rally: continue                      # between points -> ignore
    hit["rally"] = rally
    kept.append(hit)
hits = kept

MIN_RALLY_HITS = 3
keep, new_id = [], 0
for r_id in sorted({x["rally"] for x in hits}):
    grp = [x for x in hits if x["rally"] == r_id]
    if len(grp) < MIN_RALLY_HITS or len({x["player"] for x in grp}) < 2: continue
    new_id += 1
    for x in grp: x["rally"] = new_id
    keep += grp
hits, rally = keep, new_id

hd = pd.DataFrame(hits)
hd.to_csv("outputs/hits.csv", index=False)
print(hd[["rally", "frame", "time_s", "player", "serve", "dist", "dv"]].to_string() if len(hd) else "no hits found")
print("total hits:", len(hd), "| rallies:", rally)

# ---- minimap ----
S = int((h * 0.6 - 2 * PAD) / L)
def P(x, y): return (int(x*S) + PAD, int(y*S) + PAD)
base = np.full((int(L*S) + 2*PAD, int(W*S) + 2*PAD, 3), (60,120,60), np.uint8)
white = (255,255,255)
cv2.rectangle(base, P(0,0), P(W,L), white, 1)
for x in (0.46, W-0.46): cv2.line(base, P(x,0), P(x,L), white, 1)
for y in (0.76, L-0.76, L/2-1.98, L/2+1.98): cv2.line(base, P(0,y), P(W,y), white, 1)
cv2.line(base, P(W/2,0), P(W/2,L/2-1.98), white, 1)
cv2.line(base, P(W/2,L/2+1.98), P(W/2,L), white, 1)
cv2.line(base, P(0,L/2), P(W,L/2), (200,200,200), 2)

# ---- check video ----
col = {"P1": (255,128,0), "P2": (0,0,255)}
hit_at = {x["frame"]: x for x in hits}
count = {"P1": 0, "P2": 0}
last, marks, trail, cur_rally = None, [], [], 0
out = cv2.VideoWriter("outputs/hits_check.mp4", cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
i = 0
while True:
    ok, f = cap.read()
    if not ok: break
    if i in hit_at:
        hh = hit_at[i]; count[hh["player"]] += 1; last = (hh, i)
        if hh["rally"] != cur_rally: cur_rally, marks = hh["rally"], []
        marks.append((P(hh["cx"], hh["cy"]), hh["player"]))
    p = None if np.isnan(X[i]) else (int(X[i]), int(Y[i]))
    trail = (trail + [p])[-10:]
    for q in trail:
        if q: cv2.circle(f, q, 5, (0,255,255), -1)
    mm = base.copy()
    for name, g in box.items():
        if i in g.index:
            r = g.loc[i]
            cv2.rectangle(f, (int(r.x1), int(r.y1)), (int(r.x2), int(r.y2)), col[name], 2)
            cv2.circle(mm, P(r.cx, r.cy), 6, col[name], -1)
    for m, name in marks:
        cv2.drawMarker(mm, m, col[name], cv2.MARKER_CROSS, 12, 2)
    if last and i - last[1] < 20:
        hh = last[0]
        label = ("SERVE " if hh["serve"] else "HIT ") + hh["player"]
        cv2.circle(f, (int(hh["x"]), int(hh["y"])), 30, (0,255,255), 3)
        cv2.putText(f, label, (int(hh["x"]) + 35, int(hh["y"])),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 3)
    cv2.rectangle(f, (30, 185), (330, 230), (0,0,0), -1)
    cv2.putText(f, f"frame {i}", (40, 220), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
    cv2.rectangle(f, (30, 125), (560, 175), (0,0,0), -1)
    cv2.putText(f, f"Rally {cur_rally}   Hits  P1: {count['P1']}   P2: {count['P2']}", (40, 162),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
    mh, mw = mm.shape[:2]
    f[10:10+mh, w-mw-10:w-10] = mm
    out.write(f); i += 1
out.release()
print("done", i, "frames")