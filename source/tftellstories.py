
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


def MakeSimulationFolder(simulationNumber):
  foldername = path + basefoldername + str(simulationNumber)
  os.makedirs(foldername, exist_ok=True)

  return foldername

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
  print()
  predicted_token_index = np.argmax(token_predictions)
  predicted_token = layer.token_strings[predicted_token_index].numpy().decode('utf-8')
  print(f'Predicted token: "{predicted_token}" with count {token_predictions[predicted_token_index]}', flush=True)
  print()

def Run(simulationNumber: int, configuration: MultigramConfiguration, prompt: list[str]):
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


  layerSize = configuration.GetLayerSize()
  distance = configuration.GetMaxDistance()
  print(f'Running simulation {simulationNumber} with layer size {layerSize}, max distance {distance}, and configuration: {configuration.GetDescription()}')

  datafolder = MakeSimulationFolder(simulationNumber) + '/'
  layer = TFLayerModule(configuration, "TFTellStories", load_existing=True)

  print(f'Generating tokens from data folder {datafolder}')
  start_of_line_embedding = Get_embedding('>>>')
  layer(tf.constant(datafolder), tf.constant('>>>'), tf.constant(start_of_line_embedding), tf.constant(False), tf.constant(False))
  PrintTokenPredictions(layer, 2000)
  #token_predictions = layer.token_predictions.numpy()
  #for i in range(len(token_predictions)):
  #  token = token_predictions[i]
  #  if token > 20:
  #    predicted_token = layer.token_strings[i].numpy().decode('utf-8')
  #    print(f'{predicted_token}({token})', end=' ', flush=True)

  #print(token_predictions, flush=True)

  for token in prompt:
    embedding = Get_embedding(token)
    end_of_line = IsEndOfLine(token)
    print(f'Calling model with token "{token}"')
    layer(tf.constant(datafolder), tf.constant(token), tf.constant(embedding), tf.constant(end_of_line), tf.constant(False))
    PrintTokenPredictions(layer, 2000)
    if end_of_line:
      layer(tf.constant(datafolder), tf.constant('>>>'), tf.constant(start_of_line_embedding), tf.constant(False), tf.constant(False))
    print()

  story_done = False
  while not story_done:
    # Get the next token prediction from the layer.
    PrintTokenPredictions(layer, 20)
    token_predictions = layer.token_predictions.numpy()
    #print(token_predictions, flush=True)
    predicted_token_index = np.argmax(token_predictions)
    predicted_token = layer.token_strings[predicted_token_index].numpy().decode('utf-8')

    embedding = Get_embedding(predicted_token)
    end_of_line = IsEndOfLine(predicted_token)

    # Execute the tick with the predicted token and its embedding.
    layer(tf.constant(datafolder), tf.constant(predicted_token), tf.constant(embedding), tf.constant(end_of_line), tf.constant(False))

    print(predicted_token, end=' ', flush=True)

    if end_of_line:
      # If we reach the end of a line, we can decide to stop or continue based on some condition.
      # For this example, let's just stop after one line for demonstration purposes.
      story_done = True
      print()  # Print a newline after the story is done.

  # Flip the last parameter to True if we want to save the state of the layer after processing the prompt.
  layer(tf.constant(datafolder), tf.constant('>>>'), tf.constant(start_of_line_embedding), tf.constant(False), tf.constant(False))

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
    print(f'Usage: {sys.argv[0]} <simulation number> [iterations]')
    exit(0)

  simulationNumber = int(sys.argv[1])
  configfilename = MakeSimulationFolder(simulationNumber) + '/configuration.json'
  if os.path.exists(configfilename):
    with open(configfilename, 'r') as configfile:
      configuration_object = json.load(configfile)
      configuration = MultigramConfiguration('', configuration_object)
      if not configuration.valid:
        print(f'Configuration for simulation {simulationNumber} is not valid')
        exit(0)

  if len(sys.argv) > 2:
    configuration.SetIterationCount(int(sys.argv[2]))

  # Figure out a better way.
  prompt = ['They', 'also', 'helped']
  Run(simulationNumber, configuration, prompt)
