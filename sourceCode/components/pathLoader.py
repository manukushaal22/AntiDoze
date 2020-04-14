import argparse

class PathLoader:
    args = None

    def __init__(self, shapePredictorPath, alarmPath, webcamId):
        ap = argparse.ArgumentParser()
        ap.add_argument("-p", "--shape-predictor", default=shapePredictorPath,
            help="path to facial landmark predictor")
        ap.add_argument("-a", "--alarm", type=str, default=alarmPath,
            help="path alarm .WAV file")
        ap.add_argument("-w", "--webcam", type=int, default=webcamId,
            help="index of webcam on system")
        PathLoader.args = vars(ap.parse_args())

    def getShapePredictorPath(self):
        return PathLoader.args["shape_predictor"]

    def getAlarmPath(self):
        return PathLoader.args["alarm"]

    def getWebcamId(self):
        return PathLoader.args["webcam"]
