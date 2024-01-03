import numpy
import threading



class Buffer:

    def __init__(self):
        self.array = numpy.array([])

    def getSampleCount(self):
        return len(self.array)
    
    def read(self, sample_count):
        output_samples = self.array[:sample_count]
        self.array = self.array[sample_count:]
        return output_samples

    def write(self, samples):
        self.array = numpy.concatenate([self.array, samples])



class NodeInput:

    def __init__(self):
        self.connected_output = None

    def connectTo(self, node_output):
        if isinstance(node_output, BaseNode):
            node_output = list(node_output.outputs.values())[0]
        self.connected_output = node_output
        node_output.registerConsumer(self)

    def read(self, sample_count):
        if self.connected_output != None:
            return self.connected_output.read(sample_count, self)
        else:
            return numpy.zeros(sample_count)

    # convenience operator for connecting nodes
    # 'input << output' shorthand for 'input.connectTo(output)'
    # operator can be chained
    def __lshift__(self, output):
        self.connectTo(output)
        return output



class NodeOutput:

    def __init__(self, parent_node):
        self.parent_node = parent_node
        self.thread_lock = threading.Lock()
        self.buffers = {}

    def registerConsumer(self, consumer):
        self.buffers[consumer] = Buffer()

    def read(self, sample_count, consumer):
        buffer = self.buffers[consumer]

        self.thread_lock.acquire()
        while buffer.getSampleCount() < sample_count:
            self.parent_node.work(sample_count - buffer.getSampleCount())
        self.thread_lock.release()
            
        return buffer.read(sample_count)

    def write(self, samples):
        for buffer in self.buffers.values():
            buffer.write(samples)



class BaseNode:

    def __init__(self):
        self.inputs = {}
        self.outputs = {}

    def defineInputs(self, keys):
        for key in keys:
            self.inputs[key] = NodeInput()
    
    def defineOutputs(self, keys):
        for key in keys:
            self.outputs[key] = NodeOutput(self)

    # convenience methods to reduce verbosity when connecting simple nodes
    # allows 'node << node' in place of 'node.inputs["key"] << node.outputs["key"]'
    # when there is only a single input / output

    def connectTo(self, output):
        if isinstance(output, BaseNode):
            output = list(output.outputs.values())[0]
        list(self.inputs.values())[0].connectTo(output)

    def registerConsumer(self, input):
        if isinstance(input, BaseNode):
            input = list(input.inputs.values())[0]
        list(self.outputs.values())[0].registerConsumer(input)

    # convenience operator for connecting nodes
    def __lshift__(self, output):
        self.connectTo(output)
        return output



class ManualNode:

    def __init__(self, node, output_keys=None):
        self.node = node
        self.inputs = {}
        self.outputs = {}

        if output_keys == None:
            output_keys = node.outputs.keys()

        for key in node.inputs:
            self.inputs[key] = NodeOutput(self)
            node.inputs[key].connectTo(self.inputs[key])

        for key in output_keys:
            self.outputs[key] = NodeInput()
            self.outputs[key].connectTo(node.outputs[key])

    # convenience methods for reading from simple nodes
    # allow 'node.read(amt)' and 'node.write(samp)' instead of
    # 'node.outputs["key"].read(amt)' and 'node.inputs["key"].write(samp)'
    # when there is only one input / output
    
    def read(self, sample_count):
        return list(self.outputs.values())[0].read(sample_count)

    def write(self, samples):
        list(self.inputs.values())[0].write(samples)
        


class NodeGroup:

    def __init__(self):
        self.inputs = {}
        self.outputs = {}

    def mapInput(self, node_input, key):
        self.inputs[key] = node_input

    def mapOutput(self, node_output, key):
        self.output[key] = node_output
