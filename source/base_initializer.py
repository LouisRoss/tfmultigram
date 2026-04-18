import math
import numpy as np
from multigramconfiguration import MultigramConfiguration

class BaseInitializer:
  def __init__(self, configuration: MultigramConfiguration):
    self.layer_size = configuration.GetLayerSize()
    self.maxdistance = configuration.GetMaxDistance()
    self.outputwidth = configuration.GetOutputWidth()
    self.interconnectCount = configuration.GetInterconnectCount()
    self.inputwidth = self.outputwidth * self.interconnectCount

    self.xedgesize, self.yedgesize = self.GenerateSizes()

    self.inputs = []
    for input in range(self.inputwidth):
      self.inputs.append(self.layer_size-self.inputwidth+input)

    self.outputs = []
    for output in range(self.outputwidth):
      self.outputs.append(output)

    self.base = 2 * self.xedgesize
    self.Out1 = 0


  def GenerateSizes(self):
    edgesize = int(math.sqrt(self.layer_size))
    xedgesize = edgesize
    yedgesize = edgesize
    # If not a perfect square, use the smallest rectangle that fully contains all cells.
    while xedgesize * yedgesize < self.layer_size:
      yedgesize += 1

    return (xedgesize, yedgesize)

  def InitializeConnections(self):
    """ Set the internal connections between neurons in each of the single populations.
    """
    layer = np.zeros((self.maxdistance, self.layer_size, self.layer_size), dtype=np.int32)

    return layer

  def GenerateSpikes(self, duration):
    print(f'Using X edge size {self.xedgesize}, Y edge size {self.yedgesize}')


    print(f'Creating tensor with duration {duration}, thickness {self.thickness}, layer size {self.layer_size}')
    initialspikes = np.zeros((duration, self.thickness, 1, self.layer_size), dtype=np.int32)

    return initialspikes

