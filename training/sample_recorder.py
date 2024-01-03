from flow.io import AudioIO
import pathlib
import scipy.io
import numpy
from node_extras import Recorder



BLOCK_SIZE = 1024
SAMP_RATE = 16000
REC_DURATION = 2



audio = AudioIO(2, SAMP_RATE, BLOCK_SIZE)
rec = Recorder(BLOCK_SIZE)

rec << audio

audio.start()
rec.start()

while 1:
    label = input("Create sample: ")
    print("Recording ...")
    audio = rec.record(int(SAMP_RATE * REC_DURATION))
    audio = numpy.interp(audio, (-1, 1), (-32768, 32767))
    audio = audio.astype(numpy.int16)
    print("Writing to disk ...")
    folder = pathlib.Path("samples/%s/" % label)
    if not folder.is_dir():
        folder.mkdir(parents=True)
    ids = [int(f.stem) for f in folder.iterdir() if f.is_file()]
    if not ids:
        new_id = 0
    else:
        new_id = max(ids) + 1
    path = folder / ("%i.wav" % new_id)
    scipy.io.wavfile.write(str(path), SAMP_RATE, audio)
    print("Saved to %s" % str(path))
    print()