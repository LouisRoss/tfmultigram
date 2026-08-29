from multigramconfiguration import MultigramConfiguration
from initloader import InitLoader
import tensorflow as tf
import numpy as np



class TFLayerModule(tf.Module):
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


  def __init__(self, configuration: MultigramConfiguration, name: str = None, load_existing: bool = False):
    super().__init__(name=name)
    self.is_built = tf.Variable(False)
    self.load_existing = load_existing

    self.configuration = configuration

    initializers = self.configuration.GetInitializers()
    selected_initializer = self.configuration.GetSelectedInitializer()
    print(f'Initializers are {initializers}, using initializer {selected_initializer}')
    self.init_loader = InitLoader(initializers[selected_initializer], self.configuration)

    self.layer_size = self.configuration.GetLayerSize()
    self.maxdistance = self.configuration.GetMaxDistance()
    self.outputwidth = self.configuration.GetOutputWidth()
    self.interconnectCount = self.configuration.GetInterconnectCount()
    self.embedding_length = self.configuration.GetEmbeddingLength()
    self.embedding_threshold = self.configuration.GetThreshold()
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
    self.current_new_token_index = tf.Variable(0, dtype=tf.int32, name='current_new_token_index', trainable=False)
    self.token_firing = tf.Variable(tf.zeros((self.layer_size, 1), dtype=tf.int32), name='token_firing', trainable=False)
    self.token_firing_history = tf.Variable(tf.zeros((self.maxdistance, 1, self.layer_size), dtype=tf.int32), name='token_firing_history', trainable=False)


  def AcceptToken(self, token: 'str', embedding: list[float]):
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

  def FireTokens(self):
    self.token_firing.assign(self.tokens * self.maxdistance)
    self.token_firing_history.assign(tf.concat([tf.expand_dims(tf.transpose(self.token_firing), 0), self.token_firing_history[:-1]], axis=0))
    self.token_firing_history.assign(tf.maximum(tf.subtract(self.token_firing_history, 1), 0))
    expanded_firing_history = tf.broadcast_to(self.token_firing_history, [self.maxdistance, self.layer_size, self.layer_size])
    synaptic_contribution = tf.reduce_sum(expanded_firing_history * self.connections, axis=0)
    token_firing = tf.reduce_sum(synaptic_contribution, axis=1)
    return self.token_predictions.assign(token_firing) # Softmax?

  def ClearState(self):
    self.tokens.assign(tf.zeros_like(self.tokens))
    self.token_activations.assign(tf.zeros_like(self.token_activations))
    self.activeconnections.assign(tf.zeros_like(self.activeconnections))
    self.connectedhistory.assign(tf.zeros_like(self.connectedhistory))
    self.token_history.assign(tf.zeros_like(self.token_history))
    self.token_predictions.assign(tf.zeros_like(self.token_predictions))
    self.token_firing.assign(tf.zeros_like(self.token_firing))
    self.token_firing_history.assign(tf.zeros_like(self.token_firing_history))

    return self.token_predictions

  def ConnectHistory(self):
    self.connectedhistory.assign(self.activeconnections * tf.broadcast_to(self.token_history, [self.maxdistance, self.layer_size, self.layer_size]))
    self.connections.assign_add(tf.cast(tf.greater(self.connectedhistory, 0), tf.int32))

  def PredictNextToken(self):
    self.token_predictions.assign(tf.reduce_sum(tf.reduce_sum(self.activeconnections * self.connections, axis=0), axis=0))

  def PushTokenHistory(self):
    self.token_history.assign(tf.concat([tf.expand_dims(tf.transpose(self.tokens), 0), self.token_history[:-1]], axis=0))

  def ExecuteTick(self, token, embedding, end_of_line):
    self.AcceptToken(token, embedding)
    self.ForwardConnectTokens()
    self.ConnectHistory()
    #self.PredictNextToken()
    self.PushTokenHistory()
    #self.FireTokens()

    #self.token_history.assign(1 - tf.cast(end_of_line, tf.int32) * self.token_history)
    
    tf.cond(end_of_line,
      lambda: self.ClearState(),
      lambda: self.FireTokens())
    
    
    return tf.constant(0)

  def FinalizeTick(self, datafolder):
    tf.print('Finalizing tick, saving data to', datafolder)
    tf.print(self.token_strings, summarize=-1, sep=', ')

    # Serialize the tensor structure into a binary string block
    serialized_embeddings = tf.io.serialize_tensor(self.token_embeddings.read_value())
    tf.io.write_file(datafolder + 'embeddings.dat', serialized_embeddings)
    serialized_connections = tf.io.serialize_tensor(self.connections.read_value())
    tf.io.write_file(datafolder + 'connections.dat', serialized_connections)
    serialized_token_history = tf.io.serialize_tensor(self.token_history.read_value())
    tf.io.write_file(datafolder + 'token_history.dat', serialized_token_history)
    serialized_token_strings = tf.io.serialize_tensor(self.token_strings.read_value())
    tf.io.write_file(datafolder + 'token_strings.dat', serialized_token_strings)
    
    return tf.constant(0)

  @tf.function(input_signature=(tf.TensorSpec(shape=(), dtype=tf.string), tf.TensorSpec(shape=(), dtype=tf.string), tf.TensorSpec(shape=(768,), dtype=tf.float32), tf.TensorSpec(shape=(), dtype=tf.bool), tf.TensorSpec(shape=(), dtype=tf.bool)))
  def __call__(self, datafolder, token, embedding, end_of_line, log):
    # Create variables on first call.
    if not self.is_built.value():
      if self.load_existing:
        # Read the binary block and parse it back to its original data type
        serialized_embeddings = tf.io.read_file(datafolder + 'embeddings.dat')
        self.token_embeddings.assign(tf.io.parse_tensor(serialized_embeddings, out_type=tf.float32))
        empty_embeddings = tf.reduce_sum(self.token_embeddings, axis=1) == 0
        empty_embeddings_mask = tf.cast(empty_embeddings, tf.int32)
        self.current_new_token_index.assign(tf.cast(tf.argmax(empty_embeddings_mask, axis=0), dtype=tf.int32))
        serialized_connections = tf.io.read_file(datafolder + 'connections.dat')
        self.connections.assign(tf.io.parse_tensor(serialized_connections, out_type=tf.int32))
        serialized_token_history = tf.io.read_file(datafolder + 'token_history.dat')
        self.token_history.assign(tf.io.parse_tensor(serialized_token_history, out_type=tf.int32))
        serialized_token_strings = tf.io.read_file(datafolder + 'token_strings.dat')
        self.token_strings.assign(tf.io.parse_tensor(serialized_token_strings, out_type=tf.string))

        tf.print(f'Connections shape: {self.connections.shape}')
        tf.print(f'Token history shape: {self.token_history.shape}')
        tf.print(f'Token strings shape: {self.token_strings.shape}')
        tf.print(f'Token embeddings shape: {self.token_embeddings.shape}')
        tf.print(f'Current new token index:')
        tf.print(self.current_new_token_index)
        tf.print(self.token_strings)

      self.is_built.assign(True)

    tf.cond(log,
      lambda: self.FinalizeTick(datafolder),
      lambda: self.ExecuteTick(token, embedding, end_of_line))

    return self.token_predictions

