import cv2, json, numpy as np, pandas as pd

W, L, PAD = 6.1, 13.4, 20          # court metres
pts = np.array(json.load(open("court.json")), np.float32)
H = cv2.getPerspectiveTransform(pts, np.array([[0,0],[W,0],[W,L],[0,L]], np.float32))

cap = cv2.VideoCapture("clip_new.mp4")
w, h, fps, n = int(cap.get(3)), int(cap.get(4)), cap.get(5), int(cap.get(7))
S = int((h * 0.6 - 2 * PAD) / L)   # minimap pixels per metre

# --- filter to the 2 players ---
d = pd.read_csv("outputs/players.csv")
xy = cv2.perspectiveTransform(d[["feet_x","feet_y"]].values.astype(np.float32).reshape(-1,1,2), H).reshape(-1,2)
d["cx"], d["cy"] = xy[:,0], xy[:,1]
d = d[d.cx.between(-1.5, W+1.5) & d.cy.between(-3, L+3)]   # small margin around court

keep = ["feet_x","feet_y","cx","cy","x1","y1","x2","y2","lw_x","lw_y","rw_x","rw_y"]
rows = []
for fr, g in d.groupby("frame"):
    for name, half, cy0 in (("P1", g[g.cy < L/2], L/4), ("P2", g[g.cy >= L/2], 3*L/4)):
        if len(half) == 0: continue
        r = half.loc[((half.cx - W/2)**2 + (half.cy - cy0)**2).idxmin()]
        rows.append(dict(frame=fr, player=name, **{k: r[k] for k in keep}))
p = pd.DataFrame(rows)

clean = []
for name in ["P1", "P2"]:
    q = p[p.player == name].set_index("frame")[keep].reindex(range(n))
    q.index.name = "frame"
    q = q.interpolate(limit=15, limit_area="inside").rolling(5, center=True, min_periods=1).median()
    q["player"] = name
    clean.append(q.reset_index())
clean = pd.concat(clean).dropna(subset=["cx"])
clean.to_csv("outputs/players_clean.csv", index=False)
print(clean.groupby("player").size())

# --- minimap ---
def P(x, y): return (int(x*S) + PAD, int(y*S) + PAD)
base = np.full((int(L*S) + 2*PAD, int(W*S) + 2*PAD, 3), (60,120,60), np.uint8)
white = (255,255,255)
cv2.rectangle(base, P(0,0), P(W,L), white, 1)
for x in (0.46, W-0.46): cv2.line(base, P(x,0), P(x,L), white, 1)
for y in (0.76, L-0.76, L/2-1.98, L/2+1.98): cv2.line(base, P(0,y), P(W,y), white, 1)
cv2.line(base, P(W/2,0), P(W/2,L/2-1.98), white, 1)
cv2.line(base, P(W/2,L/2+1.98), P(W/2,L), white, 1)
cv2.line(base, P(0,L/2), P(W,L/2), (200,200,200), 2)

# --- check video ---
out = cv2.VideoWriter("outputs/players_clean_check.mp4", cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
by = {k: g.set_index("frame") for k, g in clean.groupby("player")}
col = {"P1": (255,128,0), "P2": (0,0,255)}
trail = {"P1": [], "P2": []}
i = 0
while True:
    ok, f = cap.read()
    if not ok: break
    mm = base.copy()
    for name, g in by.items():
        here = i in g.index
        if here:
            r = g.loc[i]
            cv2.rectangle(f, (int(r.x1), int(r.y1)), (int(r.x2), int(r.y2)), col[name], 2)
            cv2.putText(f, name, (int(r.x1), int(r.y1) - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.8, col[name], 2)
            trail[name] = (trail[name] + [P(r.cx, r.cy)])[-30:]
        for a, b in zip(trail[name], trail[name][1:]): cv2.line(mm, a, b, col[name], 1)
        if here: cv2.circle(mm, trail[name][-1], 6, col[name], -1)
    cv2.polylines(f, [pts.astype(int)], True, (0,255,255), 1)
    mh, mw = mm.shape[:2]
    f[10:10+mh, w-mw-10:w-10] = mm
    out.write(f); i += 1
out.release()
print("done", i, "frames")