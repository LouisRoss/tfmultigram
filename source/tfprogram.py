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
    self.connections = tf.Variable(self.init_loader.InitializeConnections(), name='connections', trainable=False)
    self.activeconnections = tf.Variable(np.zeros((self.maxdistance, self.layer_size, self.layer_size), dtype=np.int32), name='activeconnections', trainable=False)
    self.connectedhistory = tf.Variable(np.zeros((self.maxdistance, self.layer_size, self.layer_size), dtype=np.int32), name='connectedhistory', trainable=False)

    # self.tokens = tf.Variable(tf.zeros((1, self.layer_size, 1), dtype=tf.int32), name='tokens', trainable=False)
    self.token_delays = tf.Variable(tf.zeros((self.maxdistance, self.layer_size, 1), dtype=tf.int32), name='token_delays', trainable=False)

    # self.range_mask has a shape of (maxdistance, layer_size, 1), where the second dimension holds layer_size
    # copies of the maxdistance index: [[[1],[1],1...], [[2],[2],[2]...], ...]. 
    range_values = tf.range(1, self.maxdistance + 1)                           # [1, 2, ..., maxdistance]
    range_values = range_values[:, None, None]                                  # Reshape to (maxdistance, 1, 1) for broadcasting
    range_mask = np.full((self.maxdistance, self.layer_size, 1), range_values)  # Broadcast to fill (maxdistance, layer_size, 1)
    self.range_mask = tf.constant(range_mask, dtype=tf.int32)

    self.tokens = tf.Variable(tf.zeros((self.layer_size, 1), dtype=tf.int32), name='tokens', trainable=False)
    self.token_activations = tf.Variable(tf.zeros((self.maxdistance, self.layer_size, 1), dtype=tf.int32), name='token_activations', trainable=False)
    self.token_history = tf.Variable(tf.zeros((self.maxdistance, 1, self.layer_size), dtype=tf.int32), name='token_history', trainable=False)
    self.token_predictions = tf.Variable(tf.zeros((self.layer_size), dtype=tf.int32), name='token_predictions', trainable=False)
    self.timers = tf.Variable(tf.zeros((self.maxdistance, self.layer_size, 1), dtype=tf.int32))
    self.token_timers = tf.Variable(tf.zeros((self.maxdistance, self.layer_size, 1), dtype=tf.int32), name='token_timers', trainable=False)
    # self.token_timers = self.tokens * self.range_mask
    # self.token_delay = tf.Variable(tf.zeros((self.maxdistance, self.layer_size, 1), dtype=tf.int32))
    # self.token_delay = tf.Variable(tf.zeros((self.maxdistance, self.layer_size, 1), dtype=tf.int32))


  def PropagateTokens(self):
    #self.token_timers.assign(self.tokens * self.range_mask)
    self.token_timers.assign(self.tokens[:, None])

  def ForwardConnectTokens(self):
    # self.token_activations = tf.cast(tf.equal(self.token_timers, 1), tf.int32)
    # self.token_timers.assign(tf.maximum(tf.subtract(self.token_timers, 1,), 0))
    self.token_activations.assign(np.full((8,4,1), self.tokens))
    # self.activeconnections = self.token_activations * self.connections
    self.activeconnections.assign(np.full((8,4,4), self.token_activations))

  def ConnectHistory(self):
    self.connectedhistory.assign(self.activeconnections * np.full((8,4,4), self.token_history))
    self.connections.assign_add(tf.cast(tf.greater(self.connectedhistory, 0), tf.int32))

  def PredictNextToken(self):
    self.token_predictions.assign(tf.reduce_sum(tf.reduce_sum(self.connectedhistory, axis=0), axis=0))

  def PushTokenHistory(self):
    self.token_history.assign(tf.concat([tf.expand_dims(tf.transpose(self.tokens), 0), self.token_history[:-1]], axis=0))

  @tf.function
  def __call__(self, datafolder, log=False):
    # Create variables on first call.
    if not self.is_built:
      self.is_built = True

    # self.PropagateTokens()
    self.ForwardConnectTokens()
    self.ConnectHistory()
    self.PredictNextToken()
    self.PushTokenHistory()


    if log:
      #tf.print(self.connections, summarize=-1, sep=',', output_stream= 'file://' + datafolder + 'fullconnections.dat')
      tf.print(self.spikes, summarize=-1, sep=',', output_stream= 'file://' + datafolder + 'fullspike.dat')
      tf.print(self.potentials, summarize=-1, sep=',', output_stream= 'file://' + datafolder + 'fullactivations.dat')
      tf.print(self.hebbtimers, summarize=-1, sep=',', output_stream= 'file://' + datafolder + 'fullhebbtimers.dat')

    return self.token_predictions
  

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
