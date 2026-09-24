import os
import re
import sys
import csv
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
  max_prediction = np.max(token_predictions)
  for i in range(len(token_predictions)):
    token_count = token_predictions[i]
    if token_count == max_prediction:
      predicted_token = layer.token_strings[i].numpy().decode('utf-8')
      print(f'{predicted_token}({token_count})', end=' ', flush=True)
  predicted_token_index = np.argmax(token_predictions)
  predicted_token = layer.token_strings[predicted_token_index].numpy().decode('utf-8')
  print(f'Predicted token: "{predicted_token}" with count {token_predictions[predicted_token_index]}', flush=True)


def ExamineTokenHistory(layer: TFLayerModule, indexes: list[int]):
  """
  Examine the token history for the given indexes.
  """
  if len(indexes) == 0:
    indexes = list(range(layer.configuration.GetLayerSize()))
  token_names = [layer.token_strings[i].numpy().decode('utf-8') for i in indexes]
  token_names = [token_name for token_name in token_names if token_name != '']

  history2d = tf.squeeze(layer.token_history)
  collected_history = tf.gather(history2d, indexes, axis=1)
  history_data = []
  history_data.append(['Distance'] + token_names)
  for distance in range(layer.configuration.GetMaxDistance()):
    history_distance = collected_history[distance].numpy()
    history_data.append([f'{distance + 1}'] + list(map(str, history_distance)))

  historyfilename = GetCurrentInternalFolder() + str(runnumber) + '_token_history.csv'
  with open(historyfilename, mode='w', newline='', encoding='utf-8') as history_file:
    writer = csv.writer(history_file)
    writer.writerows(history_data)


def ExamineTokenFiringHistory(layer: TFLayerModule, indexes: list[int]):
  """
  Examine the token firing history for the given indexes.
  """
  if len(indexes) == 0:
    indexes = list(range(layer.configuration.GetLayerSize()))
  token_names = [layer.token_strings[i].numpy().decode('utf-8') for i in indexes]
  token_names = [token_name for token_name in token_names if token_name != '']

  history2d = tf.squeeze(layer.token_firing_history)
  collected_firing_history = tf.gather(history2d, indexes, axis=1)

  firing_history_data = []
  firing_history_data.append(['Distance'] + token_names)
  for distance in range(layer.configuration.GetMaxDistance()):
    firing_distance = collected_firing_history[distance].numpy()
    firing_history_data.append([f'{distance + 1}'] + list(map(str, firing_distance)))

  firinghistoryfilename = GetCurrentInternalFolder() + str(runnumber) + '_token_firing_history.csv'
  with open(firinghistoryfilename, mode='w', newline='', encoding='utf-8') as firing_history_file:
    writer = csv.writer(firing_history_file)
    writer.writerows(firing_history_data)


def ExamineConnections(layer: TFLayerModule, indexes: list[int]):
  """
  Examine the connections for the given indexes.
  """
  if len(indexes) == 0:
    indexes = list(range(layer.configuration.GetLayerSize()))
  token_names = [layer.token_strings[i].numpy().decode('utf-8') for i in indexes]
  token_names = [token_name for token_name in token_names if token_name != '']

  connection_data = []
  connection_data.append(['Distance', 'Token'] + token_names)

  distance = 1
  for layer_no in range(layer.configuration.GetMaxDistance()):
    connectionlayer = layer.connections[layer_no]
    collected_layer = tf.gather(connectionlayer, indexes, axis=1).numpy()
    for index in range(len(collected_layer)):
      firing_distance = collected_layer[index]
      connection_data.append([f'{distance}', layer.token_strings[index].numpy().decode('utf-8')] + list(map(str, firing_distance)))

  connectionfilename = GetCurrentInternalFolder() + str(runnumber) + '_connections.csv'
  with open(connectionfilename, mode='w', newline='', encoding='utf-8') as connection_file:
    writer = csv.writer(connection_file)
    writer.writerows(connection_data)


def ExamineSynapticContribution(layer: TFLayerModule, indexes: list[int]):
  """
  Examine the synaptic contribution for the given indexes.
  """
  if len(indexes) == 0:
    indexes = list(range(layer.configuration.GetLayerSize()))
  token_names = [layer.token_strings[i].numpy().decode('utf-8') for i in indexes]
  token_names = [token_name for token_name in token_names if token_name != '']

  expanded_firing_history = tf.broadcast_to(layer.token_firing_history, [layer.configuration.GetMaxDistance(), layer.configuration.GetLayerSize(), layer.configuration.GetLayerSize()])
  layer_contribution = expanded_firing_history * layer.connections
  synaptic_contribution = tf.reduce_sum(layer_contribution, axis=0)
  #synaptic_contribution_sum = tf.reduce_sum(synaptic_contribution, axis=1)
  synaptic_contribution_sum = tf.math.count_nonzero(synaptic_contribution, axis=1)
  collected_synapses = tf.gather(synaptic_contribution, indexes, axis=1)

  synaptic_contribution_data = []
  synaptic_contribution_data.append(['Distance', 'Token', 'Total'] + token_names)
  distance = 0
  for index in range(len(synaptic_contribution)):
    synaptic_contribution_row = collected_synapses[index].numpy()
    synaptic_contribution_data.append([f'{distance}', layer.token_strings[index].numpy().decode('utf-8'), str(synaptic_contribution_sum[index].numpy())] + list(map(str, synaptic_contribution_row)))

  for layer_no in range(layer.configuration.GetMaxDistance()):
    distance += 1
    layer_synaptic_contribution = layer_contribution[layer_no]
    layer_synaptic_contribution_sum = tf.reduce_sum(layer_synaptic_contribution, axis=1)
    collected_layer_synaptic_contribution = tf.gather(layer_synaptic_contribution, indexes, axis=1)

    for index in range(len(synaptic_contribution)):
      layer_synaptic_contribution_row = collected_layer_synaptic_contribution[index].numpy()
      synaptic_contribution_data.append([f'{distance}', layer.token_strings[index].numpy().decode('utf-8'), str(layer_synaptic_contribution_sum[index].numpy())] + list(map(str, layer_synaptic_contribution_row)))

  synapticcontributionfilename = GetCurrentInternalFolder() + str(runnumber) + '_synaptic_contribution.csv'
  with open(synapticcontributionfilename, mode='w', newline='', encoding='utf-8') as synaptic_file:
    writer = csv.writer(synaptic_file)
    writer.writerows(synaptic_contribution_data)


def ExamineLayerState(layer: TFLayerModule, settings: Dict):
  """
  Examine the internal state of the layer.
  """
  global runnumber

  runnumber += 1

  options = []
  if "options" in settings:
    options = settings["options"]

  indexes = []
  if "indexes" in settings:
    indexes = settings["indexes"]

  if ExaminationOption.SYNAPTIC_CONTRIBUTION in options:
    ExamineSynapticContribution(layer, indexes)
  if ExaminationOption.TOKEN_HISTORY in options:
    ExamineTokenHistory(layer, indexes)
  if ExaminationOption.TOKEN_FIRING_HISTORY in options:
    ExamineTokenFiringHistory(layer, indexes)
  if ExaminationOption.CONNECTIONS in options:
    ExamineConnections(layer, indexes)


