import math
import numpy as np
from multigramconfiguration import MultigramConfiguration

class BaseInitializer:
  def __init__(self, configuration: MultigramConfiguration):
    self.thickness = configuration.GetThickness()
    self.layer_size = configuration.GetLayerSize()
    self.threshold = configuration.GetThreshold()
    self.outputwidth = configuration.GetOutputWidth()
    self.interconnectCount = configuration.GetInterconnectCount()
    self.inputwidth = self.outputwidth * self.interconnectCount

    self.xedgesize, self.yedgesize = self.GenerateSizes()
    self.xedgethick, self.yedgethick = self.GenerateThicknesses()

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

  def GenerateThicknesses(self):
    edgesize = int(math.sqrt(self.thickness))
    xedgesize = edgesize
    yedgesize = edgesize
    # If not a perfect square, use the largest rectangle where all rows are full.
    while xedgesize * (yedgesize+1) < self.thickness:
      yedgesize += 1

    return (xedgesize, yedgesize)

  def InitializeInterconnects(self, interconnect_patterns):
    """ Generate the interconnects between populations.
        Encode in an array, with one element for each population.
        Each population element in the array is itself an array pointing
        to input populations for the element's population.
    """
    pattern = []
    if self.interconnectCount in interconnect_patterns:
      pattern = interconnect_patterns[self.interconnectCount]

    connections = np.zeros((self.thickness, len(pattern), 1), dtype=np.int32)

    i = 0
    for y in range(self.yedgethick):
      for x in range(self.xedgethick):
        patindex = 0
        pat = np.zeros((len(pattern), 1), dtype=np.int32)
        for connection in pattern:
          xsource = x + connection[0]
          if xsource < 0:
            xsource = 0
          elif xsource >= self.xedgethick:
            xsource = self.xedgethick - 1

          ysource = y + connection[1]
          if ysource < 0:
            ysource = 0
          elif ysource >= self.yedgethick:
            ysource = self.yedgethick - 1

          #print(f'i={i}, y={y}, patindex={patindex}, connection[0]={connection[0]}, connection[1]={connection[1]}')
          source = ysource * self.xedgethick + xsource
          pat[patindex] = [source]

          patindex += 1

        connections[i] = pat

        i += 1

    return connections


  def InitializeConnectionDelays(self):
    delays = np.ones((self.thickness, self.layer_size, self.layer_size), dtype=np.int32)

    return delays
  

  def InitializeConnections(self):
    """ Set the internal connections between neurons in each of the single populations.
    """
    layer = np.zeros((self.thickness, self.layer_size, self.layer_size), dtype=np.int32)

    return layer

  def GenerateSpikes(self, duration):
    print(f'Using X edge size {self.xedgesize}, Y edge size {self.yedgesize}')


    print(f'Creating tensor with duration {duration}, thickness {self.thickness}, layer size {self.layer_size}')
    initialspikes = np.zeros((duration, self.thickness, 1, self.layer_size), dtype=np.int32)

    return initialspikes

