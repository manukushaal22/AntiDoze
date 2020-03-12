from imutils.video import VideoStream
from imutils import face_utils
import imutils
import time
import dlib
import cv2

from sourceCode import pathLoader, speaker, EARcalculator

paths = pathLoader.PathLoader("shape_predictor_68_face_landmarks.dat","alarm.wav",0)
speakerSystem = speaker.Speaker(paths.getAlarmPath())
eyeAspectRatio = EARcalculator.EyeAspectRatio()

EYE_AR_THRESH = 0.3
EYE_AR_CONSEC_FRAMES = 48

COUNTER = 0

print("[INFO] loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(paths.getShapePredictorPath())

print("[INFO] starting video stream thread...")
camVideo = VideoStream(src=paths.getWebcamId()).start()
time.sleep(1.0)

while True:
    frame = camVideo.read()
    frame = imutils.resize(frame, width=450)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    rects = detector(gray, 0)

    for rect in rects:
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)
        ear = eyeAspectRatio.getTotalEyeAspectRatio(shape)

        leftEyeHull = cv2.convexHull(eyeAspectRatio.getLeftEye())
        rightEyeHull = cv2.convexHull(eyeAspectRatio.getRightEye())
        cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

        if ear < EYE_AR_THRESH:
            COUNTER += 1
            if COUNTER >= EYE_AR_CONSEC_FRAMES:
                if not speakerSystem.isPlaying():
                    speakerSystem.startAlarm()
                cv2.putText(frame, "DROWSINESS ALERT!", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            COUNTER = 0
            speakerSystem.stopAlarm()

        cv2.putText(frame, "EAR: {:.2f}".format(ear), (300, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

cv2.destroyAllWindows()
camVideo.stop()

