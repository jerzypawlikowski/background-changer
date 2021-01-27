import cv2 as cv
import numpy as np

MARGIN = 50

background = cv.imread("bg.jpg")
cascade_classifier = cv.CascadeClassifier("haarcascade_frontalface_default.xml")
video_capture = cv.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()

    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    faces = cascade_classifier.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv.CASCADE_SCALE_IMAGE,
    )

    for (x, y, w, h) in faces:
        center = (x + w // 2, y + h // 2)
        frame_bg = cv.ellipse(
            background.copy(),
            center,
            (w // 2, h // 2 + MARGIN),
            0,
            0,
            360,
            (0, 0, 0),
            -1,
        )
        mask = np.zeros(frame.shape, dtype=np.uint8)
        cv.ellipse(
            mask, center, (w // 2, h // 2 + MARGIN), 0, 0, 360, (255, 255, 255), -1
        )
        frame = (255 - (frame * mask)) + frame_bg

    if len(faces) == 0:
        frame = background

    # Display the resulting frame
    cv.imshow("Video", frame)

    if cv.waitKey(1) & 0xFF == ord("q"):
        break

# When everything is done, release the capture
video_capture.release()
cv.destroyAllWindows()
