import os
import sys
import json

from pyparsing import Dict

from multigramconfiguration import MultigramConfiguration
from base_initializer import BaseInitializer
from tflayermodule import TFLayerModule
from tfexamineinternals import ExamineLayerState, load_embeddings, Get_global_embedding, Get_token_index, IsEndOfLine, PrintTokenPredictions, ExaminationOption
import tensorflow as tf
import numpy as np


data_path ='/record/testinternal'
embedding_path = os.path.join(data_path, "embeddings.json")

fileparse = r'^([a-zA-Z]+)(\d*)$'

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

settings = {
  "options": [
    ExaminationOption.TOKEN_HISTORY,
    ExaminationOption.TOKEN_FIRING_HISTORY,
    ExaminationOption.CONNECTIONS,
    ExaminationOption.SYNAPTIC_CONTRIBUTION
  ],
  "indexes": []
}


configuration = MultigramConfiguration('', test_config)
layer = TFLayerModule(configuration, name='test_internal', load_existing=False)


def RunASequence(prompt: list[str], start_of_line: bool = False, end_of_line: bool = False):
  global layer
  global settings

  if start_of_line:
    start_of_line_embedding = Get_global_embedding('>>>')
    print(f'Calling model with token ">>>"')
    layer(tf.constant(data_path), tf.constant('>>>'), tf.constant(start_of_line_embedding), tf.constant(False), tf.constant(False))

  for token in prompt:
    embedding = Get_global_embedding(token)
    line_end = IsEndOfLine(token)
    print(f'Calling model with token "{token}"', end=' ')
    layer(tf.constant(data_path), tf.constant(token), tf.constant(embedding), tf.constant(line_end), tf.constant(False))
    PrintTokenPredictions(layer, 0)

  ExamineLayerState(layer, settings)
  if end_of_line:
    end_of_line_embedding = Get_global_embedding('.')
    print(f'Calling model with token "."', end=' ')
    layer(tf.constant(data_path), tf.constant('.'), tf.constant(end_of_line_embedding), tf.constant(True), tf.constant(False))
    PrintTokenPredictions(layer, 0)

  print()


def Run():
  """
  Run the simulation described by the given configuration.
  """
  global layer
  global configuration
  global settings

  layerSize = configuration.GetLayerSize()
  distance = configuration.GetMaxDistance()
  print(f'Running simulation with layer size {layerSize}, max distance {distance}, and configuration: {configuration.GetDescription()}')

  prompt1 = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10']
  RunASequence(prompt1, start_of_line=True, end_of_line=True)
  prompt2 = ['1', '2', '3', '10']
  RunASequence(prompt2, start_of_line=True, end_of_line=False)

  settings['indexes'] = [Get_token_index(layer, token) for token in ['>>>', '1', '2', '3', '.']]
  RunASequence(['1','2'], start_of_line=True, end_of_line=False)
  settings['indexes'] = []
  RunASequence(['3'], start_of_line=False, end_of_line=True)


# Execution starts here.
if __name__ == "__main__":
  load_embeddings()

  Run()
