import cv2, json
cap = cv2.VideoCapture("clip_new.mp4")
cap.set(cv2.CAP_PROP_POS_FRAMES, 100)
ok, f = cap.read()
pts = []
def click(e, x, y, flags, p):
    if e == cv2.EVENT_LBUTTONDOWN and len(pts) < 4:
        pts.append([x, y]); cv2.circle(f, (x, y), 6, (0, 0, 255), -1); cv2.imshow("court", f)
cv2.namedWindow("court", cv2.WINDOW_NORMAL)
cv2.imshow("court", f); cv2.setMouseCallback("court", click)
print("Click: far-left, far-right, near-right, near-left (outer doubles corners)")
while len(pts) < 4:
    if cv2.waitKey(20) == 27: break
cv2.destroyAllWindows()
json.dump(pts, open("court.json", "w"))
print(pts)