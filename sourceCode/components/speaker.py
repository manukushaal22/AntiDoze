import playsound
from threading import Thread


class Speaker:
    state, path = False, None

    def __init__(self, path):
        self.path = path

    def sound_alarm(self, path):
        playsound.playsound(path)

    def isPlaying(self):
        return self.state

    def startAlarm(self):
        self.state = True

        if Speaker.path != "":
            t = Thread(target=self.sound_alarm,
                       args=(self.path,))
            t.deamon = True
            t.start()

    def stopAlarm(self):
        self.state = False
