import cv2

cap = cv2.VideoCapture('Videos/video1.mp4')

while True:
    success, img = cap.read()
    if cv2.waitKey(1) & 0xFF == ord('q') or not success:
        break
    cv2.imshow("Video", img)