import cv2, numpy as np
def small(f): return cv2.resize(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY), (160, 90)).astype(np.float32)
c = cv2.VideoCapture("clip.mp4"); ok, ref = c.read(); ref = small(ref)
o = cv2.VideoCapture("original.mp4")
w, h, fps = int(o.get(3)), int(o.get(4)), o.get(5)
errs = []
for i in range(60):
    ok, f = o.read()
    if not ok: break
    errs.append(np.mean((small(f) - ref) ** 2))
k = int(np.argmin(errs))
print("offset k =", k, "mse =", round(errs[k], 2), "next best =", round(sorted(errs)[1], 2))
o.set(cv2.CAP_PROP_POS_FRAMES, k)
out = cv2.VideoWriter("clip_new.mp4", cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
for _ in range(984):
    ok, f = o.read()
    if not ok: break
    out.write(f)
out.release()
print("wrote clip_new.mp4")