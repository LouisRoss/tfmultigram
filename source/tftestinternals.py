import os
import sys
import json

from multigramconfiguration import MultigramConfiguration
from base_initializer import BaseInitializer
from tflayermodule import TFLayerModule
import tensorflow as tf
import numpy as np
from ollama import Client

EMPTY_EMBEDDING = [0.0] * 768  # Assuming the embedding size is 768, adjust as necessary
EMPTY_EMBEDDING[0] = 1.0  # Set the first element to 1.0 to indicate an empty embedding

OLLAMA_HOST = '192.168.1.142'
OLLAMA_PORT = 11434
OLLAMA_URL = f"http://{OLLAMA_HOST}:{OLLAMA_PORT}"
OLLAMA_MODEL = "embeddinggemma"

data_path ='/record/testinternal'
embedding_path = os.path.join(data_path, "embeddings.json")

fileparse = r'^([a-zA-Z]+)(\d*)$'

embedding_map = {}
embeddings_dirty = False

test_config = {
    "name": "Test Multigram internals",
    "description": "Multigram with layer size 20 and max distance 8.",
    "layerSize": 20,
    "maxdistance": 8,
    "embedding_length": 768,
    "threshold": 0.9,
    "interconnectCount": 1,
    "outputWidth": 2,
    "selectedInitializer": 0,
    "initializers": [
        "base_initializer"
    ]
}

configuration = MultigramConfiguration('', test_config)
layer = TFLayerModule(configuration, name='test_internal', load_existing=False)
client = Client(OLLAMA_URL)


def load_embeddings(): 
    """
    Load embeddings from a file if it exists.
    """
    global embedding_map
    if os.path.exists(embedding_path):
        print(f'Loading existing embeddings from file.')
        with open(embedding_path, "r") as f:
            embedding_map.update(json.load(f))
        print(f'Embeddings loaded: {len(embedding_map)} entries.')
    else:
        print("No embeddings file found. Starting with an empty embedding map.")


def Get_global_embedding(token: str) -> list[float]:
  global embeddings_dirty

  embedding = embedding_map.get(token, EMPTY_EMBEDDING)

  if embedding == EMPTY_EMBEDDING:
    print(f'Embedding for token "{token}" not found, requesting embedding.')
    response = client.embed(model=OLLAMA_MODEL, input=token)

    if len(response.embeddings) > 0:
        embedding = response.embeddings[0]
        embedding_map[token] = embedding
        embeddings_dirty = True

  return embedding


def Get_model_embedding(layer: TFLayerModule, token: str) -> list[float]:
  return layer.token_embeddings.get(token, EMPTY_EMBEDDING)


def IsEndOfLine(token: str) -> bool:
  """
  Check if the given token indicates the end of a line.
  """
  return token in ['.', '!', '?']


def PrintTokenPredictions(layer: TFLayerModule, threshold: int = 20):
  """
  Print the token predictions from the given layer.
  """
  token_predictions = layer.token_predictions.numpy()
  for i in range(len(token_predictions)):
    token_count = token_predictions[i]
    if token_count > threshold:
      predicted_token = layer.token_strings[i].numpy().decode('utf-8')
      print(f'{predicted_token}({token_count})', end=' ', flush=True)
  predicted_token_index = np.argmax(token_predictions)
  predicted_token = layer.token_strings[predicted_token_index].numpy().decode('utf-8')
  print(f'Predicted token: "{predicted_token}" with count {token_predictions[predicted_token_index]}', flush=True)
  print()


def RunASequence(prompt: list[str], start_of_line: bool = False, end_of_line: bool = False):
  if start_of_line:
    start_of_line_embedding = Get_global_embedding('>>>')
    print(f'Calling model with token ">>>"')
    layer(tf.constant(data_path), tf.constant('>>>'), tf.constant(start_of_line_embedding), tf.constant(False), tf.constant(False))

  for token in prompt:
    embedding = Get_global_embedding(token)
    line_end = IsEndOfLine(token)
    print(f'Calling model with token "{token}"')
    layer(tf.constant(data_path), tf.constant(token), tf.constant(embedding), tf.constant(line_end), tf.constant(False))
    PrintTokenPredictions(layer, 2000)

  if end_of_line:
    end_of_line_embedding = Get_global_embedding('.')
    print(f'Calling model with token "."')
    layer(tf.constant(data_path), tf.constant('.'), tf.constant(end_of_line_embedding), tf.constant(True), tf.constant(False))
    PrintTokenPredictions(layer, 2000)

  print()


def ExamineLayerState():
  """
  Examine the internal state of the layer.
  """
  one_index   = layer.token_strings.numpy().tolist().index(b'1')
  two_index   = layer.token_strings.numpy().tolist().index(b'2')
  three_index = layer.token_strings.numpy().tolist().index(b'3')
  four_index  = layer.token_strings.numpy().tolist().index(b'4')
  five_index  = layer.token_strings.numpy().tolist().index(b'5')
  six_index   = layer.token_strings.numpy().tolist().index(b'6')
  seven_index = layer.token_strings.numpy().tolist().index(b'7')
  eight_index = layer.token_strings.numpy().tolist().index(b'8')
  nine_index  = layer.token_strings.numpy().tolist().index(b'9')
  ten_index   = layer.token_strings.numpy().tolist().index(b'10')

  indexes = [one_index, two_index, three_index, four_index, five_index, six_index, seven_index, eight_index, nine_index, ten_index]

  for distance in range(test_config["maxdistance"]):
    print(f'Layer connections at distance {distance + 1}: ', end='')
    for i in range(len(indexes) - distance - 1):
      connection_value = layer.connections[distance, indexes[i+distance+1], indexes[i]].numpy()
      print(f'{layer.token_strings[indexes[i]].numpy().decode("utf-8")}->{layer.token_strings[indexes[i+distance+1]].numpy().decode("utf-8")}: {connection_value}', end='  ')
    print()

  connection_value = layer.connections[0, ten_index, three_index].numpy()
  print(f'Distance 1: {layer.token_strings[three_index].numpy().decode("utf-8")}->{layer.token_strings[ten_index].numpy().decode("utf-8")}: {connection_value}')
  connection_value = layer.connections[1, ten_index, two_index].numpy()
  print(f'Distance 2: {layer.token_strings[two_index].numpy().decode("utf-8")}->{layer.token_strings[ten_index].numpy().decode("utf-8")}: {connection_value}')
  connection_value = layer.connections[2, ten_index, one_index].numpy()
  print(f'Distance 3: {layer.token_strings[one_index].numpy().decode("utf-8")}->{layer.token_strings[ten_index].numpy().decode("utf-8")}: {connection_value}')

  source_sum = tf.reduce_sum(layer.connections, axis=1)
  distance_sum = tf.reduce_sum(source_sum, axis=1)
  print(f'Total connections at each distance: {distance_sum.numpy()}')

  print(f'Token history shape: {layer.token_history.shape}')
  print(f'Token history for distance 0: {layer.token_history[0].numpy()}')
  print(f'Token history for distance 1: {layer.token_history[1].numpy()}')
  print(f'Token history for distance 2: {layer.token_history[2].numpy()}')
  print(f'Token history for distance 3: {layer.token_history[3].numpy()}')
  print(f'Token history for distance 4: {layer.token_history[4].numpy()}')
  print(f'Token history for distance 5: {layer.token_history[5].numpy()}')
  print(f'Token history for distance 6: {layer.token_history[6].numpy()}')
  print(f'Token history for distance 7: {layer.token_history[7].numpy()}')

  print()
  print(f'Token Firing History shape: {layer.token_firing_history.shape}')
  print(f'Token Firing History for distance 0: {layer.token_firing_history[0].numpy()}')
  print(f'Token Firing History for distance 1: {layer.token_firing_history[1].numpy()}')
  print(f'Token Firing History for distance 2: {layer.token_firing_history[2].numpy()}')
  print(f'Token Firing History for distance 3: {layer.token_firing_history[3].numpy()}')
  print(f'Token Firing History for distance 4: {layer.token_firing_history[4].numpy()}')
  print(f'Token Firing History for distance 5: {layer.token_firing_history[5].numpy()}')
  print(f'Token Firing History for distance 6: {layer.token_firing_history[6].numpy()}')
  print(f'Token Firing History for distance 7: {layer.token_firing_history[7].numpy()}')

  expanded_firing_history = tf.broadcast_to(layer.token_firing_history, [layer.maxdistance, layer.layer_size, layer.layer_size])
  synaptic_contribution = tf.reduce_sum(expanded_firing_history * layer.connections, axis=0)
  print()
  for i in range(layer.layer_size):
    print(f'synaptic contribution for token {i+1}: {synaptic_contribution[i].numpy()}')
  print()
  token_firing = tf.reduce_sum(synaptic_contribution, axis=1)
  print(f'expanded firing history shape: {expanded_firing_history.shape}')
  print(f'synaptic contribution shape: {synaptic_contribution.shape}')
  print(f'token firing shape: {token_firing.shape}')

  print()
  print(f'Token firing:')
  print(token_firing.numpy())
  print(f'Token predictions: {layer.token_predictions.numpy()}')




def Run():
  """
  Run the simulation described by the given configuration.
  """
  layerSize = configuration.GetLayerSize()
  distance = configuration.GetMaxDistance()
  print(f'Running simulation with layer size {layerSize}, max distance {distance}, and configuration: {configuration.GetDescription()}')

  prompt1 = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10']
  RunASequence(prompt1, start_of_line=True, end_of_line=True)
  prompt2 = ['1', '2', '3', '10']
  RunASequence(prompt2, start_of_line=True, end_of_line=False)
  ExamineLayerState()

  RunASequence([], start_of_line=True, end_of_line=True)
  RunASequence(['1','2'], start_of_line=True, end_of_line=False)
  ExamineLayerState()
  RunASequence(['3'], start_of_line=False, end_of_line=False)
  ExamineLayerState()


  if embeddings_dirty:
    with open(os.path.join(data_path, "embeddings.json"), "w") as f:
      json.dump(embedding_map, f, indent=4)  # indent adds readability


# Execution starts here.
if __name__ == "__main__":
  load_embeddings()

  # Figure out a better way.
  prompt = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '.', '1', '3', '5', '7', '9', '.', '1', '2', '4', '6', '8', '10', '.', '1', '2', '3', '4', '5', '.']
  Run()
