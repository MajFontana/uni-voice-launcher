from tensorflow import keras
from rt_signal.io import AudioIO
import numpy
import tensorflow as tf
from rt_signal_extra import Recorder
import os
import global_hotkeys as ghk
import threading



python = r'C:\Users\Foxner\AppData\Local\Programs\Python\Python311\pythonw.exe "C:\Users\Foxner\AppData\Local\Programs\Python\Python311\Lib\idlelib\idle.pyw"'
notepad = r'%windir%\system32\notepad.exe'
terminal = r'wt'
documents = r'start %windir%\explorer.exe "C:\Users\Foxner\Documents"'



def stft(data):
    spec = tf.signal.stft(data, frame_length=255, frame_step=128)
    spec = tf.abs(spec)
    spec = spec[..., tf.newaxis] # image channel dimension
    return spec



MODEL_FILE = "model"
BLOCK_SIZE = 1024
SAMP_RATE = 16000
REC_DURATION = 2
CLASSES = ["python", "notepad", "terminal", "documents"]



print("Loading trained model ...")
model = keras.models.load_model(MODEL_FILE)



audio = AudioIO(2, SAMP_RATE, BLOCK_SIZE)
rec = Recorder(BLOCK_SIZE)

rec << audio

audio.start()
rec.start()

cmd_cond = threading.Condition()
def start_command():
    cmd_cond.acquire()
    cmd_cond.notify()
    cmd_cond.release()
bindings = [
    ["window + G", None, start_command, True]
]
ghk.register_hotkeys(bindings)
ghk.start_checking_hotkeys()

while 1:
    print("Waiting for command ...")
    cmd_cond.acquire()
    cmd_cond.wait()
    cmd_cond.release()
    print("Recording ...")
    audio = rec.record(int(SAMP_RATE * REC_DURATION))
    #audio = numpy.interp(audio, (-1, 1), (-32768, 32767))
    #audio = audio.astype(numpy.int16)
    indata = stft(audio)
    pred = model.predict(numpy.array([indata]))
    pred = int(tf.argmax(pred, axis=1))
    pred = CLASSES[pred]

    match pred:
        case "python":
            os.system(python)
        case "notepad":
            os.system(notepad)
        case "terminal":
            os.system(terminal)
        case "documents":
            os.system(documents)

    print(pred)
    #print()