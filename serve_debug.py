import sys, numpy as np, pandas as pd, cv2
A = int(sys.argv[1]) if len(sys.argv) > 1 else 15
B = int(sys.argv[2]) if len(sys.argv) > 2 else 100

K = 4
cap = cv2.VideoCapture("clip_new.mp4"); n = int(cap.get(7))
b = pd.read_csv("outputs/clip_ball.csv").set_index("Frame").reindex(range(n))
vis = b.Visibility == 1
xs = b.X.where(vis).astype(float).interpolate(limit=10, limit_area="inside")
ys = b.Y.where(vis).astype(float).interpolate(limit=10, limit_area="inside")
X, Y = xs.values, ys.values
vbx, vby = xs - xs.shift(K), ys - ys.shift(K)
vax, vay = xs.shift(-K) - xs, ys.shift(-K) - ys
dv = np.hypot(vax - vbx, vay - vby).values
sp = np.append(np.hypot(np.diff(X), np.diff(Y)), np.nan)

pl = pd.read_csv("outputs/players_clean.csv")
box = {k: g.set_index("frame") for k, g in pl.groupby("player")}
def bd(name, f):
    g = box[name]
    if f not in g.index or np.isnan(X[f]): return np.nan
    r = g.loc[f]
    return np.hypot(max(r.x1 - X[f], 0, X[f] - r.x2), max(r.y1 - Y[f], 0, Y[f] - r.y2))

rows = []
for f in range(A, B):
    rows.append(dict(frame=f, vis=int(vis[f]), X=X[f], Y=Y[f], speed=sp[f], dv=dv[f],
                     vy_before=vby.values[f], vy_after=vay.values[f],
                     dP1=bd("P1", f), dP2=bd("P2", f)))
d = pd.DataFrame(rows).round(0)
d.to_csv(f"outputs/debug_{A}_{B}.csv", index=False)
pd.set_option("display.width", 200)
print(d.to_string(index=False))