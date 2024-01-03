import numpy
import threading
import scipy.signal
import math
from .nodes import Buffer, BaseNode, ManualNode


class NullSink(BaseNode):

    def __init__(self, block_size):
        super().__init__()
        self.defineInputs(["samples"])

        self.block_size = block_size
        self.thread = threading.Thread(target=self._threadLoop)

    def _threadLoop(self):
        while True:
            self.inputs["samples"].read(self.block_size)

    def start(self):
        self.thread.start()



class Clock(BaseNode):

    def __init__(self, sample_rate, offset=0):
        super().__init__()
        
        self.defineOutputs(["time"])
        
        self.sample_rate = sample_rate
        self.current_time = offset

    def work(self, sample_count):
        last_sample_time = self.current_time + (sample_count - 1) / self.sample_rate
        time_points = numpy.linspace(self.current_time, last_sample_time, sample_count)
        self.outputs["time"].write(time_points)
        self.current_time += sample_count / self.sample_rate



class Oscillator(BaseNode):

    def __init__(self, frequency, sample_rate):
        super().__init__()
        self.defineOutputs(["sine"])

        self.frequency = frequency
        
        self.clock = ManualNode(Clock(sample_rate))

    def work(self, sample_count):
        time_points = self.clock.read(sample_count)
        oscillator_i = numpy.cos(time_points * 2 * numpy.pi * self.frequency)
        oscillator_q = numpy.sin(time_points * 2 * numpy.pi * self.frequency)
        oscillator = oscillator_i + 1j * oscillator_q
        self.outputs["sine"].write(oscillator)



class VariableOscillator(BaseNode):

    def __init__(self, sample_rate, coherent):
        super().__init__()
        self.defineInputs(["frequency"])
        self.defineOutputs(["sine"])

        self.coherent = coherent
        
        self.clock = ManualNode(Clock(sample_rate))

    def work(self, sample_count):
        frequency = self.inputs["frequency"].read(sample_count)
        time_points = self.clock.read(sample_count)
        if not self.coherent:
            oscillator_i = numpy.cos(time_points * 2 * numpy.pi * frequency)
            oscillator_q = numpy.sin(time_points * 2 * numpy.pi * frequency)
        oscillator = oscillator_i + 1j * oscillator_q
        self.outputs["sine"].write(oscillator)



class FrequencyModulator(BaseNode):

    def __init__(self, center_frequency, deviation, sample_rate, coherent):
        super().__init__()
        self.defineInputs(["baseband"])
        self.defineOutputs(["modulated"])

        self.center_frequency = center_frequency
        self.deviation = deviation

        self.oscillator = ManualNode(VariableOscillator(sample_rate, coherent))

    def work(self, sample_count):
        baseband = self.inputs["baseband"].read(sample_count)
        frequency = baseband * self.deviation + self.center_frequency
        self.oscillator.write(frequency)
        modulated = self.oscillator.read(sample_count)
        self.outputs["modulated"].write(modulated)



class QuadratureAmplitudeModulator(BaseNode):

    def __init__(self, carrier_frequency, sample_rate):
        super().__init__()
        self.defineInputs(["baseband 1"])
        self.defineInputs(["baseband 2"])
        self.defineOutputs(["modulated"])

        self.oscillator = ManualNode(Oscillator(carrier_frequency, sample_rate))

    def work(self, sample_count):
        baseband1 = self.inputs["baseband 1"].read(sample_count)
        baseband2 = self.inputs["baseband 2"].read(sample_count)
        in_phase = baseband1
        quadrature = baseband2 * 1j
        modulated = self.oscillator.read(sample_count) * (in_phase + quadrature)
        scaled = modulated / numpy.sqrt(2)
        self.outputs["modulated"].write(scaled)



class Deinterleaver(BaseNode):

    def __init__(self):
        super().__init__()
        self.defineInputs(["interleaved"])
        self.defineOutputs(["deinterleaved 1"])
        self.defineOutputs(["deinterleaved 2"])

    def work(self, sample_count):
        input_amount = sample_count * 2
        interleaved = self.inputs["interleaved"].read(input_amount)
        deint1 = interleaved[0:sample_count * 2 - 1:2]
        deint2 = interleaved[1:sample_count * 2:2]
        self.outputs["deinterleaved 1"].write(deint1)
        self.outputs["deinterleaved 2"].write(deint2)



class Repeat(BaseNode):

    def __init__(self, repetition_amount):
        super().__init__()
        self.defineInputs(["signal"])
        self.defineOutputs(["repeated"])

        self.repetition_amount = repetition_amount

        self.buffer = Buffer()

    def work(self, sample_count):
        input_amount = math.ceil((sample_count - self.buffer.getSampleCount()) / self.repetition_amount)
        if input_amount > 0:
            signal = self.inputs["signal"].read(input_amount)
            repeated = numpy.repeat(signal, self.repetition_amount)
            self.buffer.write(repeated)
        self.outputs["repeated"].write(self.buffer.read(sample_count))



class RandomSymbolSource(BaseNode):

    def __init__(self, symbol_count):
        super().__init__()
        self.defineOutputs(["random"])

        self.symbol_count = symbol_count

    def work(self, sample_count):
        randint = numpy.random.randint(0, self.symbol_count, sample_count)
        normalized = 2 * randint / (self.symbol_count - 1) - 1
        self.outputs["random"].write(normalized)



class AmplitudeModulator(BaseNode):

    def __init__(self, carrier_frequency, modulation_index, sample_rate):
        super().__init__()
        self.defineInputs(["baseband"])
        self.defineOutputs(["modulated"])

        self.modulation_index = modulation_index

        self.oscillator = ManualNode(Oscillator(carrier_frequency, sample_rate))

    def work(self, sample_count):
        carrier = self.oscillator.read(sample_count)
        baseband = self.inputs["baseband"].read(sample_count)
        normalized = (baseband + 1) / 2
        scaled = normalized * self.modulation_index + (1 - self.modulation_index)
        modulated = carrier * scaled
        self.outputs["modulated"].write(modulated)

        

class FrequencyShifter(BaseNode):

    def __init__(self, shift_amount, sample_rate):
        super().__init__()
        self.defineInputs(["signal"])
        self.defineOutputs(["shifted"])
        
        self.oscillator = Oscillator(-shift_amount, sample_rate)
        self.oscillator.outputs["sine"].registerConsumer(self)

    def work(self, sample_count):
        oscillator = self.oscillator.outputs["sine"].read(sample_count, self)
        original = self.inputs["signal"].read(sample_count)
        shifted = oscillator * original
        self.outputs["shifted"].write(shifted)



class LowPassFilter(BaseNode):

    def __init__(self, cutoff_frequency, transition_width, node_count, sample_rate):
        super().__init__()
        self.defineInputs(["signal"])
        self.defineOutputs(["filtered"])
        
        frequencies = [0, cutoff_frequency, cutoff_frequency + transition_width, sample_rate / 2]
        gain = [1, 1, 0, 0]
        self.coefficients = scipy.signal.firwin2(node_count, frequencies, gain, fs=sample_rate)
        self.filter_state = numpy.zeros(node_count - 1)

    def work(self, sample_count):
        signal = self.inputs["signal"].read(sample_count)
        filtered, self.filter_state = scipy.signal.lfilter(self.coefficients, 1, signal, zi=self.filter_state)
        self.outputs["filtered"].write(filtered)
