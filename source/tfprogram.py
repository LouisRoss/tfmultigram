import os
import re
import sys

from multigramconfiguration import MultigramConfiguration
from initloader import InitLoader
import tensorflow as tf
import numpy as np


path = '/record/'
basefoldername = 'simulation'
fileparse = r'^([a-zA-Z]+)(\d*)$'

def GetNextSimulationNumber():
  sims = [0]
  obj = os.scandir(path)
  for entry in obj:
    if entry.is_dir():
      parts = re.split(fileparse, entry.name)
      if parts[1] == 'simulation':
        sims.append(int(parts[2]))

  return max(sims) + 1

def MakeSimulationFolder(simulationNumber):
  foldername = path + basefoldername + str(simulationNumber)
  os.makedirs(foldername, exist_ok=True)

  return foldername


class LayerModule(tf.Module):
  """
  This class extends the Tensorflow Module class, so that any method decorated
  with the @tf.function notation will be compiled into a compute graph, on first
  execution, and all subsequent iterations will run on the compute device.
  The functor of this class implements a single tick of the spiking neural algorithm,
  including learning.
  """

def create_tf_constant(dist, size):
    # Create an array where each 'dist' index + 1 is repeated 'size' times in the second dimension
    values = np.arange(1, dist + 1)  # [1, 2, ..., dist]
    values = values[:, None, None]  # Reshape to (dist, 1, 1) for broadcasting
    arr = np.full((dist, size, 1), values)  # Broadcast to fill (dist, size, 1)
    return tf.constant(arr, dtype=tf.int32)  # Adjust dtype as needed


  def __init__(self, configuration: MultigramConfiguration, initializer, name=None):
    super().__init__(name=name)
    self.is_built = False

    self.configuration = configuration
    self.init_loader = initializer

    self.layer_size = self.configuration.GetLayerSize()
    self.maxdistance = self.configuration.GetMaxDistance()
    self.outputwidth = self.configuration.GetOutputWidth()
    self.interconnectCount = configuration.GetInterconnectCount()
    self.inputwidth = self.outputwidth * self.interconnectCount
    self.tick = tf.Variable(0)
    self.tflayer_size = tf.constant(self.layer_size, dtype=tf.int32)
    #self.connections = tf.Variable(self.init_loader.InitializeConnections(), name='connections', trainable=False)
    self.connections = tf.Variable(tf.zeros((self.maxdistance, self.layer_size, self.layer_size), dtype=tf.int32))
    self.tokens = tf.Variable(tf.zeros((1, self.layer_size, 1), dtype=tf.int32), name='tokens', trainable=False)
    self.token_delays = tf.Variable(tf.zeros((self.maxdistance, self.layer_size, 1), dtype=tf.int32), name='token_delays', trainable=False)

    # self.range_mask has a shape of (maxdistance, layer_size, 1), where the second dimension holds layer_size
    # copies of the maxdistance index: [[[1],[1],1...], [[2],[2],[2]...], ...]. 
    range_values = tf.range(1, self.maxdistance + 1)                           # [1, 2, ..., maxdistance]
    range_values = range_values[:, None, None]                                  # Reshape to (maxdistance, 1, 1) for broadcasting
    range_mask = np.full((self.maxdistance, self.layer_size, 1), range_values)  # Broadcast to fill (maxdistance, layer_size, 1)
    self.range_mask = tf.constant(range_mask, dtype=tf.int32)

    self.tokens = tf.Variable(tf.zeros((1, self.layer_size), dtype=tf.int32), name='tokens', trainable=False)
    self.token_timers = tf.Variable(tf.zeros((self.maxdistance, self.layer_size, 1), dtype=tf.int32), name='token_timers', trainable=False)
    self.timers = tf.Variable(tf.zeros((self.maxdistance, self.layer_size, 1), dtype=tf.int32))
    # self.token_timers = self.tokens * self.range_mask
    self.token_delay = tf.Variable(tf.zeros((self.maxdistance, self.layer_size, 1), dtype=tf.int32))
    x = (self.token_delays + self.tokens) * (self.token_delays + self.timers)

    self.connection_delays = tf.Variable(self.init_loader.InitializeConnectionDelays(), name='connection_delays', trainable=False)
    self.connection_timers = tf.Variable(tf.zeros((self.thickness, self.delaydepth, self.layer_size, self.layer_size), dtype=tf.int32), name='connection_timers', trainable=False)
    self.connection_post_timers = tf.Variable(tf.zeros((self.thickness, self.layer_size, self.layer_size), dtype=tf.int32), name='connection_post_timers', trainable=False)
    self.activeconnections = tf.Variable(tf.zeros((self.thickness, self.layer_size, self.layer_size), dtype=tf.int32), name='activeconnections', trainable=False)
    self.post_time_delay = tf.constant(25)

    self.interconnects = tf.constant(self.init_loader.InitializeInterconnects(), name='interconnects')
    self.delaytimes = tf.Variable(tf.zeros([self.thickness, 1, self.layer_size], dtype=tf.dtypes.int32), dtype=tf.dtypes.int32, name='delaytimes', trainable=False)
    self.delayguards = tf.Variable(tf.ones([self.thickness, 1, self.layer_size], dtype=tf.dtypes.int32), dtype=tf.dtypes.int32, name='delayguards', trainable=False)

  def __call__(self, datafolder, log=False):
    # Create variables on first call.
    if not self.is_built:
      self.potentials = tf.Variable(tf.zeros([self.thickness, 1, self.layer_size], dtype=tf.dtypes.int32), dtype=tf.dtypes.int32, name='potentials', trainable=False)
      self.decayedpotentials = tf.Variable(tf.zeros([self.thickness, 1, self.layer_size], dtype=tf.dtypes.int32), dtype=tf.dtypes.int32, name='decayedpotentials', trainable=False)
      self.resets = tf.Variable(tf.zeros([self.thickness, 1, self.layer_size], dtype=tf.dtypes.int32), dtype=tf.dtypes.int32, name='resets', trainable=False)
      self.hebbtimers = tf.Variable(tf.zeros([self.thickness, 1, self.layer_size], dtype=tf.dtypes.int32), dtype=tf.dtypes.int32, name='hebbtimers', trainable=False)
      initialspikes = np.zeros((self.thickness, 1, self.layer_size), dtype=np.int32)
      self.spikes = tf.Variable(tf.cast(initialspikes, tf.dtypes.int32), trainable=False)
      self.dummyspikes = tf.Variable(tf.ones([self.thickness, 1, self.layer_size], dtype=tf.int32), name='dummy_spikes', trainable=False)

      self.is_built = True



    if log:
      #tf.print(self.connections, summarize=-1, sep=',', output_stream= 'file://' + datafolder + 'fullconnections.dat')
      tf.print(self.spikes, summarize=-1, sep=',', output_stream= 'file://' + datafolder + 'fullspike.dat')
      tf.print(self.potentials, summarize=-1, sep=',', output_stream= 'file://' + datafolder + 'fullactivations.dat')
      tf.print(self.hebbtimers, summarize=-1, sep=',', output_stream= 'file://' + datafolder + 'fullhebbtimers.dat')

    return self.spikes
  

def Run(configuration: MultigramConfiguration):
  """
  Run the simulation described by the given configuration.
  """
  #tf.debugging.set_log_device_placement(True)

  simulationNumber = GetNextSimulationNumber()
  datafolder = MakeSimulationFolder(simulationNumber) + '/'
  configuration.Save(datafolder)

  layerSize = configuration.GetLayerSize()
  thickness = configuration.GetThickness()
  iterationCount = configuration.GetIterationCount()

  print(f'Running simulation {simulationNumber} with layer size {layerSize}, thickness {thickness}, iteration count {iterationCount}')


# Execution starts here.
if __name__ == "__main__":
  if len(sys.argv) < 2:
    print(f'Usage: {sys.argv[0]} <configuration> [initializer number] [iterations] [layersize] [thickness]')
    exit(0)

  configuration = MultigramConfiguration(sys.argv[1])
  if not configuration.valid:
    print(f'Configuration {sys.argv[1]} is not valid')
    exit(0)

  if len(sys.argv) > 2:
    initializer = int(sys.argv[2])
    if initializer >= len(configuration.GetInitializers()):
      print(f'Initializer {initializer} is bigger than allowed by configuration {sys.argv[1]}, which has {len(configuration.GetInitializers())} initializers')
      exit(0)

    configuration.SetSelectedInitializer(initializer)

  if len(sys.argv) > 3:
    configuration.SetIterationCount(int(sys.argv[3]))

  if len(sys.argv) > 4:
    configuration.SetIterationCount(int(sys.argv[4]))

  if len(sys.argv) > 5:
    configuration.SetLayerSize(int(sys.argv[5]))

  if len(sys.argv) > 6:
    configuration.SetThickness(int(sys.argv[6]))

  Run(configuration)
