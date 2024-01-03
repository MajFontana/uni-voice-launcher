import threading
from rt_signal.nodes import BaseNode, Buffer



class Recorder(BaseNode):

    def __init__(self, block_size):
        super().__init__()
        self.defineInputs(["audio"])
        self.block_size = block_size
        self.recording_done = threading.Condition()
        self.thread = threading.Thread(target=self._threadLoop)
        self.record_count = 0
        self.buffer = Buffer()

    def work(self, sample_count):
        samples = self.inputs["audio"].read(sample_count)
        if self.record_count > 0:
            trimmed = samples[:self.record_count]
            self.buffer.write(trimmed)
            self.record_count -= len(trimmed)
            if self.record_count == 0:
                self.recording_done.acquire()
                self.recording_done.notify()
                self.recording_done.release()

    def _threadLoop(self):
        while True:
            self.work(self.block_size)
    
    def start(self):
        self.thread.start()

    def record(self, sample_count):
        self.recording_done.acquire()
        self.record_count = sample_count
        self.recording_done.wait()
        samples = self.buffer.read(sample_count)
        self.recording_done.release()
        return samples
