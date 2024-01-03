import numpy
import pyaudio
from .nodes import BaseNode
import threading



class AudioIO(BaseNode):
    
    def __init__(self, sample_width, sample_rate, block_size):
        super().__init__()
        self.defineInputs(["audio"])
        self.defineOutputs(["audio"])

        self.sample_rate = sample_rate
        if sample_width == 1:
            self.data_type = numpy.uint8
        elif sample_width == 2:
            self.data_type = numpy.int16
        elif sample_width == 4:
            self.data_type = numpy.int32
            
        self.new_data_condition = threading.Condition()
        self.pyaudio = pyaudio.PyAudio()
        self.stream = self.pyaudio.open(
            format=self.pyaudio.get_format_from_width(sample_width),
            channels=1,
            rate=sample_rate,
            input=True,
            output=True,
            start=False,
            frames_per_buffer=block_size,
            stream_callback=self._IOCallback
            )
        
    def _IOCallback(self, binary_in, frame_count, time_info, status):
        type_info = numpy.iinfo(self.data_type)
    
        real_in = numpy.frombuffer(binary_in, dtype=self.data_type)
        normalized_in = numpy.interp(real_in, [type_info.min, type_info.max], [-1, 1])
        self.outputs["audio"].write(normalized_in)
        with self.new_data_condition:
            self.new_data_condition.notify()

        normalized_out = self.inputs["audio"].read(frame_count)
        real_out = numpy.interp(normalized_out.real, [-1, 1], [type_info.min, type_info.max])
        binary_out = real_out.round().astype(self.data_type).tobytes()
        
        return (binary_out, pyaudio.paContinue)

    def start(self):
        self.stream.start_stream()

    def work(self, sample_count):
        with self.new_data_condition:
            self.new_data_condition.wait()
