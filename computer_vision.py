import cv2

font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 0.7
color = (0,255,255)
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame. Exiting...")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(20, 20))

    for i,(x, y, w, h) in enumerate(faces):
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv2.putText(frame, f"face {i + 1}", (x - 5, y - 5), font, font_scale, color, 2, cv2.LINE_AA)
    
    cv2.putText(frame, f"Detected faces: {len(faces)}", (20,50), font, font_scale, color, 2, cv2.LINE_AA)
    cv2.putText(frame, "Press 'q' to exit", (20,20), font, 0.5, (0,0,255), 1, cv2.LINE_AA)
    cv2.imshow('Face Detection', frame)
    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
