from imutils import face_utils
from scipy.spatial import distance as dist


class EyeAspectRatio:
    lStart, lEnd, rStart, rEnd, shape = 0, 0, 0, 0, None

    def __init__(self):
        (self.lStart, self.lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
        (self.rStart, self.rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

    def calculateSingleEyeAspectRatio(self, eye):
        A = dist.euclidean(eye[1], eye[5])  # vertical
        B = dist.euclidean(eye[2], eye[4])  # vertical
        C = dist.euclidean(eye[0], eye[3])  # horizontal
        return (A + B) / (2.0 * C)

    def getTotalEyeAspectRatio(self, shape):
        self.shape = shape
        leftEAR = self.calculateSingleEyeAspectRatio(self.getLeftEye())
        rightEAR = self.calculateSingleEyeAspectRatio(self.getRightEye())
        return (leftEAR + rightEAR) / 2.0

    def getLeftEye(self):
        if self.shape is not None:
            return self.shape[self.lStart:self.lEnd]
        return None

    def getRightEye(self):
        if self.shape is not None:
            return self.shape[self.rStart:self.rEnd]
        return None
