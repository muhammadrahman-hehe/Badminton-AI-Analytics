import cv2, pandas as pd
df = pd.read_csv("outputs/clip_ball.csv").set_index("Frame")
cap = cv2.VideoCapture("clip_new.mp4")
w, h, fps = int(cap.get(3)), int(cap.get(4)), cap.get(5)
out = cv2.VideoWriter("outputs/shuttle_check.mp4", cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
trail, i = [], 0
while True:
    ok, f = cap.read()
    if not ok: break
    p = None
    if i in df.index and df.loc[i, "Visibility"] == 1:
        p = (int(df.loc[i, "X"]), int(df.loc[i, "Y"]))
    trail = (trail + [p])[-12:]
    for q in trail:
        if q: cv2.circle(f, q, 6, (0, 255, 255), -1)
    out.write(f); i += 1
out.release()
print("frames written:", i)