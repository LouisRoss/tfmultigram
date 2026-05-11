import os
import re
import sys

from multigramconfiguration import MultigramConfiguration
from initloader import InitLoader
from tokenbase import TokenBase
from tokensourcedataset import TokenSourceDataset
from ollama import Client
import tensorflow as tf
import numpy as np

OLLAMA_HOST = '192.168.1.142'
OLLAMA_PORT = 11434
OLLAMA_URL = f"http://{OLLAMA_HOST}:{OLLAMA_PORT}"
OLLAMA_MODEL = "embeddinggemma"

EMPTY_EMBEDDING = [0.0] * 768  # Assuming the embedding size is 768, adjust as necessary
EMPTY_EMBEDDING[0] = 1.0  # Set the first element to 1.0 to indicate an empty embedding


path = '/record/multigram/'
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
  client = Client(OLLAMA_URL)

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
    self.embedding_length = configuration.GetEmbeddingLength()
    self.embedding_threshold = configuration.GetThreshold()
    self.inputwidth = self.outputwidth * self.interconnectCount
    self.tick = tf.Variable(0)
    self.tflayer_size = tf.constant(self.layer_size, dtype=tf.int32)
    self.connections = tf.Variable(self.init_loader.InitializeConnections(), name='connections', trainable=False)
    self.activeconnections = tf.Variable(np.zeros((self.maxdistance, self.layer_size, self.layer_size), dtype=np.int32), name='activeconnections', trainable=False)
    self.connectedhistory = tf.Variable(np.zeros((self.maxdistance, self.layer_size, self.layer_size), dtype=np.int32), name='connectedhistory', trainable=False)

    self.tokens = tf.Variable(tf.zeros((self.layer_size, 1), dtype=tf.int32), name='tokens', trainable=False)
    self.token_activations = tf.Variable(tf.zeros((self.maxdistance, self.layer_size, 1), dtype=tf.int32), name='token_activations', trainable=False)
    self.token_history = tf.Variable(tf.zeros((self.maxdistance, 1, self.layer_size), dtype=tf.int32), name='token_history', trainable=False)
    self.token_predictions = tf.Variable(tf.zeros((self.layer_size), dtype=tf.int32), name='token_predictions', trainable=False)
    self.token_embeddings = tf.Variable(tf.zeros([self.layer_size, self.embedding_length], dtype=tf.float32), name='token_embeddings', trainable=False)
    self.token_strings = tf.Variable(tf.zeros((self.layer_size), dtype=tf.string), name='token_strings', trainable=False)
    self.current_new_token_index = tf.Variable(0)


  def AcceptToken(self, token: str):
    response = LayerModule.client.embed(model=OLLAMA_MODEL, input=token)

    # If no embeddings are returned, use an empty embedding
    embedding = EMPTY_EMBEDDING
    if len(response.embeddings) > 0:
        embedding = response.embeddings[0]

    self.tokens.assign(tf.zeros_like(self.tokens))

    similarities = tf.tensordot(self.token_embeddings, embedding, axes=1)
    token_seen = tf.cast(tf.greater(similarities, self.embedding_threshold), tf.int32)
    token_possible = tf.tensor_scatter_nd_update(token_seen, [[self.current_new_token_index]], [1])
    next_token_index = tf.cast(tf.argmax(token_possible, axis=0), dtype=tf.int32)

    if tf.less(next_token_index, self.layer_size - 1):
      self.current_new_token_index.assign_add(tf.cast(tf.equal(self.current_new_token_index, next_token_index), tf.int32))  # Increment index only if token is new

      self.token_embeddings.assign(tf.tensor_scatter_nd_update(self.token_embeddings, [[next_token_index]], [embedding]))
      self.tokens.assign(tf.tensor_scatter_nd_update(self.tokens, [[next_token_index, 0]], [1]))
      self.token_strings.assign(tf.tensor_scatter_nd_update(self.token_strings, [[next_token_index]], [token]))

  def ForwardConnectTokens(self):
    self.token_activations.assign(tf.broadcast_to(self.tokens, [self.maxdistance, self.layer_size, 1]))
    self.activeconnections.assign(tf.broadcast_to(self.token_activations, [self.maxdistance, self.layer_size, self.layer_size]))

  def ConnectHistory(self):
    self.connectedhistory.assign(self.activeconnections * tf.broadcast_to(self.token_history, [self.maxdistance, self.layer_size, self.layer_size]))
    self.connections.assign_add(tf.cast(tf.greater(self.connectedhistory, 0), tf.int32))

  def PredictNextToken(self):
    self.token_predictions.assign(tf.reduce_sum(tf.reduce_sum(self.activeconnections * self.connections, axis=0), axis=0))

  def PushTokenHistory(self):
    self.token_history.assign(tf.concat([tf.expand_dims(tf.transpose(self.tokens), 0), self.token_history[:-1]], axis=0))

  @tf.function
  def __call__(self, datafolder, token_source, log=False):
    # Create variables on first call.
    if not self.is_built:
      self.is_built = True

    while token_source.IsInputAvailable():
      token = token_source.GetNext()

      print(f'Processing token: {token.token_raw} into data folder {datafolder}')
      self.AcceptToken(token.token_raw)
      self.ForwardConnectTokens()
      self.ConnectHistory()
      self.PredictNextToken()
      self.PushTokenHistory()

      if token.end_of_line:
        self.token_history.assign(tf.zeros_like(self.token_history))


    tf.print(self.connections, summarize=-1, sep=',', output_stream= 'file://' + datafolder + 'connections.dat')
    tf.print(self.tokens, summarize=-1, sep=',', output_stream= 'file://' + datafolder + 'tokens.dat')
    tf.print(self.token_history, summarize=-1, sep=',', output_stream= 'file://' + datafolder + 'token_history.dat')
    tf.print(self.token_strings, summarize=-1, sep=',', output_stream= 'file://' + datafolder + 'token_strings.dat')
    tf.print(self.token_embeddings, summarize=-1, sep=',', output_stream= 'file://' + datafolder + 'token_embeddings.dat')

    return self.token_predictions

def MakeLayer(configuration: MultigramConfiguration):#
  initializers = configuration.GetInitializers()
  selected_initializer = configuration.GetSelectedInitializer()
  print(f'Initializers are {initializers}, using initializer {selected_initializer}')
  initializer = InitLoader(initializers[selected_initializer], configuration)
  return LayerModule(configuration, initializer)

def Run(configuration: MultigramConfiguration):
  """
  Run the simulation described by the given configuration.
  """
  #tf.debugging.set_log_device_placement(True)

  simulationNumber = GetNextSimulationNumber()
  datafolder = MakeSimulationFolder(simulationNumber) + '/'
  configuration.Save(datafolder)

  layerSize = configuration.GetLayerSize()
  distance = configuration.GetMaxDistance()
  print(f'Running simulation {simulationNumber} with layer size {layerSize}, max distance {distance}, and configuration: {configuration.GetDescription()}')

  layer = MakeLayer(configuration)

  with TokenSourceDataset("roneneldan/TinyStories", 10) as token_source:
      layer(datafolder, token_source, log=False)


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
