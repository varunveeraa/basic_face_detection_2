# Left and Right Eye Detection

# Importing the library
import cv2

# Loading the cascades
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
righteye_cascade = cv2.CascadeClassifier('haarcascade_righteye_2splits.xml')
lefteye_cascade = cv2.CascadeClassifier('haarcascade_lefteye_2splits.xml')
nose_cascade = cv2.CascadeClassifier('haarcascade_nose2.xml')
mouth_cascade = cv2.CascadeClassifier('haarcascade_mouth.xml')

# Defining a function that will do the detections
def detect(gray, frame):
    faces = face_cascade.detectMultiScale(gray, 1.3, 10)
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        righteye = righteye_cascade.detectMultiScale(roi_gray, 1.1, 30)
        for (rx, ry, rw, rh) in righteye:
            cv2.rectangle(roi_color, (rx, ry), (rx+rw, ry+rh), (255, 255, 255), 1)
        lefteye = lefteye_cascade.detectMultiScale(roi_gray, 1.1, 30)
        for (lx, ly, lw, lh) in lefteye:
            cv2.rectangle(roi_color, (lx, ly), (lx+lw, ly+lh), (255, 255, 0), 1)
        nose = nose_cascade.detectMultiScale(roi_gray, 1.1, 30)
        for (nx, ny, nw, nh) in nose:
            cv2.rectangle(roi_color, (nx, ny), (nx+nw, ny+nh), (255, 0, 0), 1)
        mouth = mouth_cascade.detectMultiScale(roi_gray, 1.1, 80)
        for (mx, my, mw, mh) in mouth:
            cv2.rectangle(roi_color, (mx, my), (mx+mw, my+mh), (0, 0, 255), 1)
    return frame

# Doing some Face Recognition with the webcam
video_capture = cv2.VideoCapture(0)
while True:
    _, frame = video_capture.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    canvas = detect(gray, frame)
    cv2.imshow('Video', canvas)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
video_capture.release()
cv2.destroyAllWindows()

#WHITE RECTANGLE FOR RIGHT EYE
#YELLOW RECTANGLE FOR LEFT EYE
#RED RECTANGLE FOR MOUTH
#BLUE RECTANGLE FOR NOSE
