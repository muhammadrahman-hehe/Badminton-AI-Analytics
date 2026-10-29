import cv2, pandas as pd
from ultralytics import YOLO

model = YOLO("yolov8n-pose.pt")
rows = []
cap = cv2.VideoCapture("clip_new.mp4")
w, h, fps = int(cap.get(3)), int(cap.get(4)), cap.get(5)
out = cv2.VideoWriter("outputs/players_check.mp4", cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
i = 0
while True:
    ok, f = cap.read()
    if not ok: break
    r = model.track(f, persist=True, tracker="botsort.yaml", imgsz=1280, conf=0.3, verbose=False)[0]
    if r.boxes.id is not None:
        ids = r.boxes.id.int().tolist()
        kp = r.keypoints.xy.cpu().numpy()
        kc = r.keypoints.conf.cpu().numpy()
        bx = r.boxes.xyxy.cpu().numpy()
        for j, tid in enumerate(ids):
            fa = [kp[j][k] for k in (15, 16) if kc[j][k] > 0.3]
            if not fa: continue
            fx = sum(p[0] for p in fa) / len(fa)
            fy = sum(p[1] for p in fa) / len(fa)
            rows.append(dict(frame=i, id=tid, feet_x=fx, feet_y=fy,
                             lw_x=kp[j][9][0], lw_y=kp[j][9][1],
                             rw_x=kp[j][10][0], rw_y=kp[j][10][1],
                             x1=bx[j][0], y1=bx[j][1], x2=bx[j][2], y2=bx[j][3]))
            cv2.circle(f, (int(fx), int(fy)), 6, (0, 255, 0), -1)
            cv2.putText(f, f"id{tid}", (int(bx[j][0]), int(bx[j][1]) - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    out.write(f); i += 1
    if i % 50 == 0: print(i, "frames")
out.release()
pd.DataFrame(rows).to_csv("outputs/players.csv", index=False)
print("done", i, "frames,", len(rows), "detections")