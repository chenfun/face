import cv2

cap = cv2.VideoCapture(0)  # 默認攝像頭
if not cap.isOpened():
    print("無法開啟攝像頭")
else:
    print("攝像頭已開啟")
    while True:
        ret, frame = cap.read()
        if ret:
            cv2.imshow("Camera", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    cap.release()
    cv2.destroyAllWindows()