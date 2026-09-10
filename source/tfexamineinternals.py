import os
import re
import sys
import json
from enum import Enum
from typing import Dict

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


internal_path = '/record/multigram/'
basefoldername = 'internal'
foldername = ''
runnumber = 0
fileparse = r'^([a-zA-Z]+)(\d*)$'


data_path ='/record/testinternal'
embedding_path = os.path.join(data_path, "embeddings.json")

fileparse = r'^([a-zA-Z]+)(\d*)$'

embedding_map = {}
client = Client(OLLAMA_URL)

class ExaminationOption(Enum):
    TOKEN_HISTORY = 1
    TOKEN_FIRING_HISTORY = 2
    CONNECTIONS = 3
    SYNAPTIC_CONTRIBUTION = 4


def GetNextExaminationNumber():
  internals = [0]
  obj = os.scandir(internal_path)
  for entry in obj:
    if entry.is_dir():
      parts = re.split(fileparse, entry.name)
      if parts[1] == 'internal':
        internals.append(int(parts[2]))

  return max(internals) + 1

def MakeInternalFolder():
  global foldername
  ExaminationNumber = GetNextExaminationNumber()
  foldername = internal_path + basefoldername + str(ExaminationNumber) + '/'
  os.makedirs(foldername, exist_ok=True)


def GetCurrentInternalFolder():
  """
  Get the current internal folder based on the latest examination number.
  """
  global foldername
  if foldername == '':
    MakeInternalFolder()
  return foldername

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

  return embedding


def Get_token_index(layer: TFLayerModule, token: str) -> int:
  """
  Get the index of the given token in the layer's token strings.
  """
  try:
    return layer.token_strings.numpy().tolist().index(bytearray(token, 'utf-8'))
  except ValueError:
    print(f'Token "{token}" not found in token strings.')
    return -1

def Get_token_string(layer: TFLayerModule, index: int) -> str:
  """
  Get the token string corresponding to the given index in the layer's token strings.
  """
  if 0 <= index < len(layer.token_strings):
    return layer.token_strings[index].numpy().decode('utf-8')
  else:
    print(f'Index {index} is out of bounds for token strings.')
    return ''
  
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


def ExamineTokenHistory(layer: TFLayerModule, indexes: list[int]):
  """
  Examine the token history for the given indexes.
  """
  #for distance in range(layer.configuration.GetMaxDistance()):
  #  print(f'Token history at distance {distance + 1}: ', end='')
  #  print(layer.token_history[distance].numpy())
  #print()
  if len(indexes) == 0:
    indexes = list(range(layer.configuration.GetLayerSize()))
  token_names = [layer.token_strings[i].numpy().decode('utf-8') for i in indexes]

  history2d = tf.squeeze(layer.token_history)
  collected_history = tf.gather(history2d, indexes, axis=1)

  historyfilename = GetCurrentInternalFolder() + str(runnumber) + '_token_history.csv'
  with open(historyfilename, 'a') as history_file:
    header = ',' + ','.join(token_names)
    history_file.write(header + '\n')  # Write the header
    for distance in range(layer.configuration.GetMaxDistance()):
      history_distance = collected_history[distance].numpy()
      history_file.write(f'Distance {distance + 1},' + ','.join(map(str, history_distance)) + '\n')
    history_file.write('\n')  # Add a newline to separate layers
  

def ExamineTokenFiringHistory(layer: TFLayerModule, indexes: list[int]):
  """
  Examine the token firing history for the given indexes.
  """
  #for distance in range(layer.configuration.GetMaxDistance()):
  #  print(f'Token firing history at distance {distance + 1}: ', end='')
  #  print(layer.token_firing_history[distance].numpy())
  #print()
  if len(indexes) == 0:
    indexes = list(range(layer.configuration.GetLayerSize()))
  token_names = [layer.token_strings[i].numpy().decode('utf-8') for i in indexes]

  history2d = tf.squeeze(layer.token_firing_history)
  collected_firing_history = tf.gather(history2d, indexes, axis=1)

  firinghistoryfilename = GetCurrentInternalFolder() + str(runnumber) + '_token_firing_history.csv'
  with open(firinghistoryfilename, 'a') as firing_history_file:
    header = ',' + ','.join(token_names)
    firing_history_file.write(header + '\n')  # Write the header
    for distance in range(layer.configuration.GetMaxDistance()):
      firing_distance = collected_firing_history[distance].numpy()
      firing_history_file.write(f'Distance {distance + 1},' + ','.join(map(str, firing_distance)) + '\n')
    firing_history_file.write('\n')  # Add a newline to separate layers


def ExamineConnections(layer: TFLayerModule, indexes: list[int]):
  """
  Examine the connections for the given indexes.
  """
  #print(f'Layer connections at distance {distance + 1}: ')
  #print(layer.connections[distance].numpy())
  #print()

  #source_sum = tf.reduce_sum(layer.connections, axis=1)
  #distance_sum = tf.reduce_sum(source_sum, axis=1)
  #print(f'Total connections at each distance: {distance_sum.numpy()}')
  #print()

  if len(indexes) == 0:
    indexes = list(range(layer.configuration.GetLayerSize()))
  token_names = [layer.token_strings[i].numpy().decode('utf-8') for i in indexes]

  connectionfilename = GetCurrentInternalFolder() + str(runnumber) + '_connections.csv'
  with open(connectionfilename, 'a') as connection_file:
    #for layer_no in range(layer.configuration.GetMaxDistance()):
    #  connectionlayer = layer.connections[layer_no]
    #  collected_layer = tf.gather(connectionlayer, indexes, axis=1)
    #  np.savetxt(connection_file, collected_layer.numpy(), header=','.join(token_names), delimiter=',', fmt='%d')
    #  connection_file.write('\n')  # Add a newline to separate layers

    header = ',' + ','.join(token_names)
    for layer_no in range(layer.configuration.GetMaxDistance()):
      connection_file.write(f'Distance {layer_no + 1}' + '\n')
      connection_file.write(header + '\n')  # Write the header
      connectionlayer = layer.connections[layer_no]
      collected_layer = tf.gather(connectionlayer, indexes, axis=1).numpy()

      for index in range(len(collected_layer)):
        firing_distance = collected_layer[index]
        connection_file.write(f'{layer.token_strings[index].numpy().decode("utf-8")},' + ','.join(map(str, firing_distance)) + '\n')


def ExamineSynapticContribution(layer: TFLayerModule, indexes: list[int]):
  """
  Examine the synaptic contribution for the given indexes.
  """
  #expanded_firing_history = tf.broadcast_to(layer.token_firing_history, [layer.configuration.GetMaxDistance(), layer.configuration.GetLayerSize(), layer.configuration.GetLayerSize()])
  #synaptic_contribution = tf.reduce_sum(expanded_firing_history * layer.connections, axis=0)
  #print()

  #for i in range(layer.configuration.GetLayerSize()):
  #  print(f'synaptic contribution for token {i+1}: {synaptic_contribution[i].numpy()}')
  #print()

  #token_firing = tf.reduce_sum(synaptic_contribution, axis=1)
  #print(f'expanded firing history shape: {expanded_firing_history.shape}')
  #print(f'synaptic contribution shape: {synaptic_contribution.shape}')
  #print(f'token firing shape: {token_firing.shape}')

  #print()
  #print(f'Token firing: {token_firing.numpy()}')

  if len(indexes) == 0:
    indexes = list(range(layer.configuration.GetLayerSize()))
  token_names = [layer.token_strings[i].numpy().decode('utf-8') for i in indexes]

  expanded_firing_history = tf.broadcast_to(layer.token_firing_history, [layer.configuration.GetMaxDistance(), layer.configuration.GetLayerSize(), layer.configuration.GetLayerSize()])
  synaptic_contribution = tf.reduce_sum(expanded_firing_history * layer.connections, axis=0)
  collected_synapses = tf.gather(synaptic_contribution, indexes, axis=1)
  print(f'Synaptic contribution shape: {synaptic_contribution.shape}')

  synapticcontributionfilename = GetCurrentInternalFolder() + str(runnumber) + '_synaptic_contribution.csv'
  with open(synapticcontributionfilename, 'a') as synaptic_file:
    #np.savetxt(synaptic_file, synaptic_contribution.numpy(), header=','.join(token_names), delimiter=',', fmt='%d')
    #synaptic_file.write('\n')  # Add a newline to separate layers

    synaptic_contribution_sum = tf.reduce_sum(synaptic_contribution, axis=1)
    #np.savetxt(synaptic_file, source_sum.numpy(), header=','.join(token_names), delimiter=',', fmt='%d')
    header = ','.join(token_names)
    synaptic_file.write(',Total,' + header + '\n')  # Write the header
    for index in range(len(synaptic_contribution)):
      synaptic_contribution_row = collected_synapses[index].numpy()
      synaptic_file.write(f'{layer.token_strings[index].numpy().decode("utf-8")},' + str(synaptic_contribution_sum[index].numpy()) + ',' + ','.join(map(str, synaptic_contribution_row)) + '\n')



def ExamineLayerState(layer: TFLayerModule, settings: Dict):
  """
  Examine the internal state of the layer.
  """
  global runnumber

  one_index   = Get_token_index(layer, '1')
  two_index   = Get_token_index(layer, '2')
  three_index = Get_token_index(layer, '3')
  four_index  = Get_token_index(layer, '4')
  five_index  = Get_token_index(layer, '5')
  six_index   = Get_token_index(layer, '6')
  seven_index = Get_token_index(layer, '7')
  eight_index = Get_token_index(layer, '8')
  nine_index  = Get_token_index(layer, '9')
  ten_index   = Get_token_index(layer, '10')

  runnumber += 1

  options = []
  if "options" in settings:
    options = settings["options"]

  indexes = []
  if "indexes" in settings:
    indexes = settings["indexes"]

  # Test, remove
  indexes = [one_index, two_index, three_index, four_index, five_index, six_index, seven_index, eight_index, nine_index, ten_index]


  if ExaminationOption.SYNAPTIC_CONTRIBUTION in options:
    ExamineSynapticContribution(layer, indexes)
  if ExaminationOption.TOKEN_HISTORY in options:
    ExamineTokenHistory(layer, indexes)
  if ExaminationOption.TOKEN_FIRING_HISTORY in options:
    ExamineTokenFiringHistory(layer, indexes)
  if ExaminationOption.CONNECTIONS in options:
    ExamineConnections(layer, indexes)


