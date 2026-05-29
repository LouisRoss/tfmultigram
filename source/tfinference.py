import os
import re
import sys
import json
import random

from multigramconfiguration import MultigramConfiguration
import tensorflow as tf
import numpy as np

path = '/record/multigram/'
basefoldername = 'simulation'

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


  def __init__(self, configuration: MultigramConfiguration, name=None):
    super().__init__(name=name)
    self.is_built = False

    self.configuration = configuration

    self.layer_size = self.configuration.GetLayerSize()
    self.maxdistance = self.configuration.GetMaxDistance()
    self.outputwidth = self.configuration.GetOutputWidth()
    self.interconnectCount = configuration.GetInterconnectCount()
    self.embedding_length = configuration.GetEmbeddingLength()
    self.embedding_threshold = configuration.GetThreshold()
    self.inputwidth = self.outputwidth * self.interconnectCount
    self.connections = tf.Variable(tf.zeros((self.maxdistance, self.layer_size, self.layer_size), dtype=tf.int32), name='connections', trainable=False)
    self.token_history = tf.Variable(tf.zeros((self.maxdistance, 1, self.layer_size), dtype=tf.int32), name='token_history', trainable=False)
    self.token_strings = tf.Variable(tf.zeros((self.layer_size), dtype=tf.string), name='token_strings', trainable=False)
    tf.print(f'Connections shape: {self.connections.shape}')

  @tf.function
  def __call__(self, datafolder, log=False):
    # Create variables on first call.
    if not self.is_built:
      self.is_built = True

    print(f'Processing multigram from data folder {datafolder}')
    # Read the binary block and parse it back to its original data type
    serialized_connections = tf.io.read_file(datafolder + 'connections.dat')
    self.connections.assign(tf.io.parse_tensor(serialized_connections, out_type=tf.int32))
    serialized_token_history = tf.io.read_file(datafolder + 'token_history.dat')
    self.token_history.assign(tf.io.parse_tensor(serialized_token_history, out_type=tf.int32))
    serialized_token_strings = tf.io.read_file(datafolder + 'token_strings.dat')
    self.token_strings.assign(tf.io.parse_tensor(serialized_token_strings, out_type=tf.string))

    tf.print(f'Connections shape: {self.connections.shape}')
    tf.print(f'Token history shape: {self.token_history.shape}')
    tf.print(f'Token strings shape: {self.token_strings.shape}')
    tf.print(self.token_strings)

    return self.token_strings

def FindAllFirstWords(layer: LayerModule):
  """
  Finds all first words in the layer, which are the tokens that have a connection from the first distance.
  """
  first_word_gate = tf.reduce_sum(layer.connections[0], axis=1)
  first_words = tf.boolean_mask(layer.token_strings, first_word_gate)
  
  return first_word_gate

def PrintPredictions(layer: LayerModule, words: list[str], distance: int, word_index: int):
  """
  Prints the predicted words in the layer, which are the tokens that have a connection from any distance.
  """
  print(f"Predictions for word '{words[word_index]}' at distance {distance}:", end=' ')
  word_predictions = tf.slice(layer.connections, begin=[distance, 0, word_index], size=[1, layer.layer_size, 1])
  for i in range(layer.layer_size):
    if word_predictions[0][i][0] > 0:
      print(f"'{words[i]}'({word_predictions[0][i][0]})", end=' ')
  print()

def FindBestNextToken(layer: LayerModule, words: list[str], token_history: list[int]) -> int:
  """
  Find the best next token based on the current token's relationships.
  This function uses the token's connections to determine the next likely token.
  NOTE: words is provided only for debugging, all logic is based on token indices.
  """
  likely_tokens = {}

  history_length = len(token_history) if len(token_history) < layer.maxdistance else layer.maxdistance
  for distance in range(1, history_length + 1):
    token = token_history[-distance]
    PrintPredictions(layer, words, distance - 1, token)
    distance_multiplier = (history_length - distance + 1) / history_length
    #distance_multiplier = 1.0

    pruned_tokens = {}
    word_predictions = tf.slice(layer.connections, begin=[distance-1, 0, token], size=[1, layer.layer_size, 1])
    for i in range(layer.layer_size):
      following_token = i
      if word_predictions[0][i][0] != 0:
        if distance > 1 and following_token in likely_tokens:
          pruned_tokens[following_token] = likely_tokens[following_token] + int(word_predictions[0][i][0].numpy()) * distance_multiplier
        elif distance == 1 and following_token not in likely_tokens:
          likely_tokens[following_token] = int(word_predictions[0][i][0].numpy()) * distance_multiplier

    if distance > 1:
      likely_tokens = dict(pruned_tokens)     # copy contents, not reference.

  print()
  print('*************************************************')
  print(f'Finding next likely token out of {len(likely_tokens)} possibilities after ' + ' -> '.join(words[token_history[i]] for i in range(0, len(token_history))))
  for token, strength in likely_tokens.items():
      print(f"  Possible next token: '{words[token]}' with strength {strength}")
  print('*************************************************')
  print()

  likely_token = max(likely_tokens, key=likely_tokens.get, default=None)
  if likely_token is not None:
      max_strength = likely_tokens[likely_token] if likely_token is not None else 0
      print(f'Most likely next token for "{words[token_history[-1]]}" is "{words[likely_token] if likely_token is not None else "<None>"}" with strength {max_strength}')

  return likely_token


def FindMostLikelyNextToken(layer: LayerModule, words: list[str], token_history: list[int]) -> int:
  """
    Find the most likely next token based on the current token's relationships.
    This function uses the token's connections to determine the next likely token.
  NOTE: words is provided only for debugging, all logic is based on token indices.
  """
  likely_tokens = {}

  history_length = len(token_history) if len(token_history) < layer.maxdistance else layer.maxdistance
  for distance in range(1, history_length + 1):
    token = token_history[-distance]
    distance_multiplier = history_length - distance + 1

    word_predictions = tf.slice(layer.connections, begin=[distance-1, 0, token], size=[1, layer.layer_size, 1])
    for i in range(layer.layer_size):
      following_token = i
      if word_predictions[0][i][0] != 0:
        if distance > 1 and following_token in likely_tokens:
          likely_tokens[following_token] += int(word_predictions[0][i][0].numpy()) * distance_multiplier
        elif distance == 1 and following_token not in likely_tokens:
          likely_tokens[following_token] = int(word_predictions[0][i][0].numpy()) * distance_multiplier

    print()
    print('*************************************************')
    print(f'Finding next likely token out of {len(likely_tokens)} possibilities after ' + ' -> '.join(words[token_history[i]] for i in range(0, len(token_history))))
    for token, strength in likely_tokens.items():
        print(f"  Possible next token: '{words[token]}' with strength {strength}")
    print('*************************************************')
    print()

    likely_token = max(likely_tokens, key=likely_tokens.get, default=None)
    if likely_token is not None:
        max_strength = likely_tokens[likely_token] if likely_token is not None else 0
        print(f'Most likely next token for "{words[token_history[-1]]}" is "{words[likely_token] if likely_token is not None else "<None>"}" with strength {max_strength}')

    return likely_token



def GenerateLikelyString(layer: LayerModule, words: list['str'], token: int) -> 'str':
  """
  Generate a likely string based on the token and its relationships.
  This function uses the token's connections to build a string representation.
  """
  print(f'Generating likely string starting with token "{words[token]}"')
  result = []

  while words[token] != '.':
    result.append(token)
    #print(words[token])
    token = FindBestNextToken(layer, words, result)

  string_result = [words[t] if t != None else '<None>' for t in result]
  return ' -> '.join(string_result)


def GenerateBestFitString(layer: LayerModule, words: list['str'], tokens: list['str']) -> 'str':
  """
  Generate the best fit string based on the token and its relationships.
  This function uses the token's connections to build a string representation.
  """
  result = [words.index(t) for t in tokens if t in words]

  found_end = False
  while not found_end:
    token = FindBestNextToken(layer, words, result)
    result.append(token)
    if token is None or words[token] == '.':
      found_end = True

  string_result = [words[t] if t != None else '<None>' for t in result]
  return ' -> '.join(string_result)


def GenerateRandomSentence(layer: LayerModule, words: list[str]) -> str:
  first_words = []
  word_predictions = tf.slice(layer.connections, begin=[0, 0, 0], size=[1, layer.layer_size, 1])
  for i in range(layer.layer_size):
    if word_predictions[0][i][0] > 0:
      first_words.append(i)

  print(f'First words: {[words[i] for i in first_words]}')
  random_root = random.choice(first_words) if len(first_words) > 0 else None

  if random_root is None:
      return "No starting token found."
  return GenerateLikelyString(layer, words, random_root)


def Run(configuration: MultigramConfiguration, datafolder: str):
  """
  Run the simulation inference described by the given configuration.
  """
  #tf.debugging.set_log_device_placement(True)

  layerSize = configuration.GetLayerSize()
  distance = configuration.GetMaxDistance()
  print(f'Examining simulation with layer size {layerSize}, max distance {distance}, and configuration: {configuration.GetDescription()}')

  layer = LayerModule(configuration)
  layer(datafolder, log=False)

  connections = layer.connections.numpy()
  #print(f'Distance 1 connections: {connections[0]}')
  """
  word_0_count = tf.reduce_sum(tf.slice(layer.connections, begin=[0, 0, 0], size=[layer.maxdistance, 1, layer.layer_size]), axis=0)
  print(f'Word 0 count: {word_0_count}')
  word_1_count = tf.slice(layer.connections, begin=[0, 0, 0], size=[1, layer.layer_size, 1])
  print(f'>>> plus 1: {tf.transpose(word_1_count[0])[0]}')
  word_2_count = tf.slice(layer.connections, begin=[1, 0, 7], size=[1, layer.layer_size, 1])
  print(f'Lily plus 2: {tf.transpose(word_2_count[0])[0]}')
  word_3_count = tf.slice(layer.connections, begin=[2, 0, 7], size=[1, layer.layer_size, 1])
  print(f'Lily plus 3: {tf.transpose(word_3_count[0])[0]}')
  """

  token_strings = layer.token_strings.numpy().tolist()
  token_strings = [s.decode('utf-8') for s in token_strings]
  print(f'Token strings: {token_strings}')

  """
  for i in range(layer.layer_size):
    if token_strings[i] != '':
      for distance in range(layer.maxdistance):
        PrintPredictions(layer, token_strings, distance, i)
  """
  PrintPredictions(layer, token_strings, 0, 0)

  #sentence = GenerateRandomSentence(layer, token_strings)
  #print(f'Generated sentence: {sentence}')

  #sentence = GenerateBestFitString(layer, token_strings, ['>>>', 'good', 'fuel', 'made'])
  #sentence = GenerateBestFitString(layer, token_strings, ['>>>', 'Can', 'you', 'share', 'it', 'with', 'me'])
  sentence = GenerateBestFitString(layer, token_strings, ['>>>', 'Can', 'you', 'share'])
  print(f'Best fit sentence: {sentence}')

# Execution starts here.
if __name__ == "__main__":
  if len(sys.argv) < 2:
    print(f'Usage: {sys.argv[0]} <simulation>')
    exit(0)

  datafolder = path + basefoldername + sys.argv[1] + '/' 
  if not os.path.exists(datafolder):
    print(f'Simulation folder {datafolder} does not exist')
    exit(0)

  with open(datafolder + 'configuration.json', 'r') as configfile:
    configuration_json = json.load(configfile)
    configuration = MultigramConfiguration("", configuration_json)
    if not configuration.valid:
      print(f'Configuration {sys.argv[1]} is not valid')
      exit(0)

  Run(configuration, datafolder)
