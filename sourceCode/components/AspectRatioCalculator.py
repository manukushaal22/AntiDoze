from imutils import face_utils
from scipy.spatial import distance as dist


class AspectRatioCalculator:
    lStart, lEnd, rStart, rEnd, shape = 0, 0, 0, 0, None

    def __init__(self):
        (self.lStart, self.lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
        (self.rStart, self.rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
        (self.mouthStart, self.mouthEnd) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]

    def setFaceShape(self,shape):
        self.shape = shape

    def calculateSingleEyeAspectRatio(self, eye):
        vertical_1 = dist.euclidean(eye[1], eye[5])  # vertical
        vertical_2 = dist.euclidean(eye[2], eye[4])  # vertical
        horizontal = dist.euclidean(eye[0], eye[3])  # horizontal
        return (vertical_1 + vertical_2) / (2.0 * horizontal)

    def getEyeAspectRatio(self):
        if self.shape is not None:
            leftEAR = self.calculateSingleEyeAspectRatio(self.getLeftEye())
            rightEAR = self.calculateSingleEyeAspectRatio(self.getRightEye())
            return (leftEAR + rightEAR) / 2.0
        return None

    def getMouthAspectRatio(self):
        inner_mouth = self.getMouth()[-8:]
        vertical_1 = dist.euclidean(inner_mouth[1], inner_mouth[7])
        vertical_2 = dist.euclidean(inner_mouth[2], inner_mouth[6])
        vertical_3 = dist.euclidean(inner_mouth[3], inner_mouth[5])
        horizontal = dist.euclidean(inner_mouth[0], inner_mouth[4])
        return (vertical_1 + vertical_2 + vertical_3) / (3.0 * horizontal)

    def getLeftEye(self):
        if self.shape is not None:
            return self.shape[self.lStart:self.lEnd]
        return None

    def getRightEye(self):
        if self.shape is not None:
            return self.shape[self.rStart:self.rEnd]
        return None

    def getMouth(self):
        if self.shape is not None:
            return self.shape[self.mouthStart:self.mouthEnd]
        return None

    def getInnerMouth(self):
        ''' last 8 points of mouth cover the inner mouth'''
        if self.shape is not None:
            return self.getMouth()[-8:]
        return None