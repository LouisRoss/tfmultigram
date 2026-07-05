
import os
import sys
import json
import re
import tensorflow as tf
import numpy as np
from ollama import Client
from tokensourcedataset import TokenSourceDataset
from multigramconfiguration import MultigramConfiguration
from tflayermodule import TFLayerModule


OLLAMA_HOST = '192.168.1.142'
OLLAMA_PORT = 11434
OLLAMA_URL = f"http://{OLLAMA_HOST}:{OLLAMA_PORT}"
OLLAMA_MODEL = "embeddinggemma"

EMPTY_EMBEDDING = [0.0] * 768  # Assuming the embedding size is 768, adjust as necessary
EMPTY_EMBEDDING[0] = 1.0  # Set the first element to 1.0 to indicate an empty embedding


path = '/record/multigram/'
basefoldername = 'simulation'
fileparse = r'^([a-zA-Z]+)(\d*)$'

embeddings_dirty = False
embedding_map = {}
client = Client(OLLAMA_URL)


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



def Run(configuration: MultigramConfiguration):
  """
  Run the simulation described by the given configuration.
  """
  #tf.debugging.set_log_device_placement(True)
  # Load dictionary from a file
  if os.path.exists("/record/embeddings.json"):
    print(f'Loading existing embeddings from file.')
    with open("/record/embeddings.json", "r") as f:
      embedding_map.update(json.load(f))

  print(f'Embeddings loaded: {len(embedding_map)} entries.')


  simulationNumber = GetNextSimulationNumber()
  datafolder = MakeSimulationFolder(simulationNumber) + '/'
  configuration.Save(datafolder)

  layerSize = configuration.GetLayerSize()
  distance = configuration.GetMaxDistance()
  print(f'Running simulation {simulationNumber} with layer size {layerSize}, max distance {distance}, and configuration: {configuration.GetDescription()}')

  layer = TFLayerModule(configuration)

  with TokenSourceDataset("roneneldan/TinyStories", 200) as token_source:
    print(f'Processing tokens into data folder {datafolder}')
    start_of_line_embedding = Get_embedding('>>>')
    layer(tf.constant(datafolder), tf.constant('>>>'), tf.constant(start_of_line_embedding), tf.constant(False), tf.constant(False))
    while token_source.IsInputAvailable():
      token = token_source.GetNext()
      print(f'{token.token_raw}', end=' ')
      embedding = Get_embedding(token.token_raw)
      layer(tf.constant(datafolder), tf.constant(token.token_raw), tf.constant(embedding), tf.constant(token.end_of_line), tf.constant(False))
      if token.end_of_line:
        layer(tf.constant(datafolder), tf.constant('>>>'), tf.constant(start_of_line_embedding), tf.constant(False), tf.constant(False))
        print()

  layer(tf.constant(datafolder), tf.constant('>>>'), tf.constant(start_of_line_embedding), tf.constant(False), tf.constant(True))

  if embeddings_dirty:
    with open("/record/embeddings.json", "w") as f:
      json.dump(embedding_map, f, indent=4)  # indent adds readability

def Get_embedding(token: str) -> list[float]:
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
