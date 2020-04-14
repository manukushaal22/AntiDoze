from imutils.video import VideoStream
from imutils import face_utils
import imutils
import time
import dlib
import cv2

from components import speaker, pathLoader, AspectRatioCalculator
import ApplicationConstants

def main():
    paths = pathLoader.PathLoader(r"../resources/shape_predictor_68_face_landmarks.dat", "../resources/alarm.wav", 0)
    speakerSystem = speaker.Speaker(paths.getAlarmPath())
    aspectRatioEstimator = AspectRatioCalculator.AspectRatioCalculator()
    constants = ApplicationConstants.Constants()

    print("[INFO] loading facial landmark predictor...")
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(paths.getShapePredictorPath())

    print("[INFO] starting video stream thread...")
    camVideo = VideoStream(src=paths.getWebcamId()).start()
    time.sleep(1.0)

    EYE_FRAME_COUNTER, MOUTH_FRAME_COUNTER, YAWN_FRAME_COUNTER, YAWN_COUNTER = 0, 0, 0, 0
    ALERT, ALERT_COUNTER = False, 0
    while True:
        frame = camVideo.read()
        frame = imutils.resize(frame, width=constants.WINDOW_SIZE)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        rects = detector(gray, 0)

        for rect in rects:
            shape = predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)
            aspectRatioEstimator.setFaceShape(shape)

            if constants.EYE_ANALYSIS_ON:
                ear = aspectRatioEstimator.getEyeAspectRatio()
                if ear < constants.EYE_AR_THRESHOLD:
                    EYE_FRAME_COUNTER += 1
                    if EYE_FRAME_COUNTER >= constants.EYE_CLOSE_CONSECUTIVE_FRAMES_THRESHOLD:
                        if constants.ALARM_SYSTEM_ACTIVATE and not speakerSystem.isPlaying():
                            speakerSystem.startAlarm()
                        if constants.WINDOW_GUI_ACTIVATE:
                            cv2.putText(frame, "DROWSINESS ALERT! (EYES)", (230, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                else:
                    EYE_FRAME_COUNTER = 0
                    if constants.ALARM_SYSTEM_ACTIVATE:
                        speakerSystem.stopAlarm()
                if constants.WINDOW_GUI_ACTIVATE:
                    cv2.putText(frame, "EAR: {:.2f}".format(ear), (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            if constants.MOUTH_ANALYSIS_ON:
                mar = aspectRatioEstimator.getMouthAspectRatio()
                if mar > constants.MOUTH_AR_THRESHOLD:
                    MOUTH_FRAME_COUNTER += 1
                    if MOUTH_FRAME_COUNTER >= constants.MOUTH_OPEN_CONSECUTIVE_FRAMES_THRESHOLD:
                        YAWN_COUNTER +=1
                        MOUTH_FRAME_COUNTER = 0
                else:
                    MOUTH_FRAME_COUNTER = 0
                    if constants.ALARM_SYSTEM_ACTIVATE:
                        speakerSystem.stopAlarm()
                if constants.WINDOW_GUI_ACTIVATE:
                    cv2.putText(frame, "MAR: {:.2f}".format(mar), (constants.WINDOW_SIZE - 150, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    cv2.putText(frame, "YAWNS: {}".format(YAWN_COUNTER), (constants.WINDOW_SIZE - 150, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                if YAWN_COUNTER >= constants.MOUTH_YAWN_PER_MINUTE_THRESHOLD:
                    if constants.ALARM_SYSTEM_ACTIVATE and not speakerSystem.isPlaying():
                        speakerSystem.startAlarm()
                        speakerSystem.stopAlarm()
                    if constants.WINDOW_GUI_ACTIVATE:
                        ALERT = True
                    YAWN_FRAME_COUNTER, YAWN_COUNTER = 0, 0
                if ALERT:
                    ALERT_COUNTER += 1
                    cv2.putText(frame, "DROWSINESS ALERT! (MOUTH)", (230, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    if ALERT_COUNTER >= 5 * constants.FRAMES_PER_SECOND:
                        ALERT = False
                        ALERT_COUNTER = 0
                if YAWN_COUNTER>0:
                    YAWN_FRAME_COUNTER += 1
                    if YAWN_FRAME_COUNTER >= 60 * constants.FRAMES_PER_SECOND:
                        YAWN_FRAME_COUNTER, YAWN_COUNTER = 0, 0

        if constants.WINDOW_GUI_ACTIVATE:
            drawEyeMouthContours(frame, aspectRatioEstimator, constants.EYE_ANALYSIS_ON, constants.MOUTH_ANALYSIS_ON)
            cv2.imshow("Frame", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

    cv2.destroyAllWindows()
    camVideo.stop()

if __name__=="__main__":

    def drawEyeMouthContours(frame, aspectRatioEstimator, isEyeOn, isMouthOn):
        if isEyeOn:
            leftEyeHull = cv2.convexHull(aspectRatioEstimator.getLeftEye())
            rightEyeHull = cv2.convexHull(aspectRatioEstimator.getRightEye())
            cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
            cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)
        if isMouthOn:
            innerMouthHull = cv2.convexHull(aspectRatioEstimator.getInnerMouth())
            cv2.drawContours(frame, [innerMouthHull], -1, (0, 255, 0), 1)

    main()