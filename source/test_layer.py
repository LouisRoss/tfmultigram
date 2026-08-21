import os
import json

import pytest

from multigramconfiguration import MultigramConfiguration
from base_initializer import BaseInitializer
from tflayermodule import TFLayerModule
import tensorflow as tf
import numpy as np

EMPTY_EMBEDDING = [0.0] * 768  # Assuming the embedding size is 768, adjust as necessary
EMPTY_EMBEDDING[0] = 1.0  # Set the first element to 1.0 to indicate an empty embedding


path = '/record/'
embedding_path = '/record/embeddings.json'
layer_path = '/record/testsimulation/'

fileparse = r'^([a-zA-Z]+)(\d*)$'

embedding_map = {}

test_config = {
    "name": "Test Multigram configuration",
    "description": "Multigram with layer size 4 and max distance 8.",
    "layerSize": 4,
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


@pytest.fixture(scope="module")
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

    yield embedding_map  # Yield the embedding map for use in tests


def Get_global_embedding(token: str) -> list[float]:
  return embedding_map.get(token, EMPTY_EMBEDDING)

def Get_model_embedding(layer: TFLayerModule, token: str) -> list[float]:
  return layer.token_embeddings.get(token, EMPTY_EMBEDDING)


@pytest.fixture
def setup_layer():
  global configuration

  configfilename = os.path.join(layer_path, 'configuration.json')
  if os.path.exists(configfilename):
    with open(configfilename, 'r') as configfile:
      configuration_object = json.load(configfile)
      configuration = MultigramConfiguration('', configuration_object)
      if not configuration.valid:
        print(f'Test configuration {configfilename} is not valid')

  layer = TFLayerModule(configuration, name='test_layer', load_existing=True)

  yield layer


class TestFullLayer:
    @pytest.mark.filterwarnings("ignore:DeprecationWarning")
    def test_basic(self, load_embeddings, setup_layer):
        embeddings = load_embeddings
        layer = setup_layer

        assert layer.tokens.shape == (configuration.GetLayerSize(), 1)
        assert layer.connections.shape == (configuration.GetMaxDistance(), configuration.GetLayerSize(), configuration.GetLayerSize())

    @pytest.mark.filterwarnings("ignore:DeprecationWarning")
    def test_connect_they_also(self, load_embeddings, setup_layer):
        embeddings = load_embeddings
        layer = setup_layer

        # Model loads from disk on first call.
        start_of_line_embedding = Get_global_embedding('>>>')
        layer(tf.constant(layer_path), tf.constant('>>>'), tf.constant(start_of_line_embedding), tf.constant(False), tf.constant(False))
        assert layer.token_strings.numpy().tolist()[0] == b'>>>', "The first token should be '>>>'"
        assert layer.token_strings.numpy().tolist()[1] == b'One', "The second token should be 'One'"
        she_index = layer.token_strings.numpy().tolist().index(b'she')
        put_index = layer.token_strings.numpy().tolist().index(b'put')
        on_index = layer.token_strings.numpy().tolist().index(b'on')
        her_index = layer.token_strings.numpy().tolist().index(b'her')
        pajamas_index = layer.token_strings.numpy().tolist().index(b'pajamas')

        # Check that the connections are set up correctly
        assert layer.connections[0, put_index, she_index].numpy() == 8, "Connection from 'they' to 'also' should be set."
        assert layer.connections[0, on_index, put_index].numpy() == 3, "Connection from 'also' to 'helped' should be set."
        assert layer.connections[0, her_index, on_index].numpy() == 7, "Connection from 'also' to 'helped' should be set."
        assert layer.connections[0, pajamas_index, her_index].numpy() == 1, "Connection from 'also' to 'helped' should be set."

        assert layer.connections[1, on_index, she_index].numpy() == 3, "Connection from 'they' to 'helped' should be set."
        assert layer.connections[1, her_index, put_index].numpy() == 3, "Connection from 'they' to 'helped' should be set."
        assert layer.connections[1, pajamas_index, on_index].numpy() == 1, "Connection from 'they' to 'helped' should be set."

        assert layer.connections[2, her_index, she_index].numpy() == 11, "Connection from 'they' to 'helped' should be set."
        assert layer.connections[2, pajamas_index, put_index].numpy() == 1, "Connection from 'they' to 'helped' should be set."

    @pytest.mark.filterwarnings("ignore:DeprecationWarning")
    def test_activate_first_word(self, load_embeddings, setup_layer):
        embeddings = load_embeddings
        layer = setup_layer

        # Model loads from disk on first call.
        start_of_line_embedding = Get_global_embedding('>>>')
        layer(tf.constant(layer_path), tf.constant('>>>'), tf.constant(start_of_line_embedding), tf.constant(False), tf.constant(False))

        start_of_line_index = layer.token_strings.numpy().tolist().index(b'>>>')
        she_index = layer.token_strings.numpy().tolist().index(b'she')

        she_embedding = layer.token_embeddings.numpy().tolist()[she_index]
        layer(tf.constant(layer_path), tf.constant('she'), tf.constant(she_embedding), tf.constant(False), tf.constant(False))

        #assert she_index == 0
        assert layer.token_history[1, 0, start_of_line_index].numpy() == 1
        assert layer.token_history[0, 0, she_index].numpy() == 1

    @pytest.mark.filterwarnings("ignore:DeprecationWarning")
    def test_activate_five_words(self, load_embeddings, setup_layer):
        embeddings = load_embeddings
        layer = setup_layer

        # Model loads from disk on first call.
        start_of_line_embedding = Get_global_embedding('>>>')
        layer(tf.constant(layer_path), tf.constant('>>>'), tf.constant(start_of_line_embedding), tf.constant(False), tf.constant(False))

        start_of_line_index = layer.token_strings.numpy().tolist().index(b'>>>')
        she_index = layer.token_strings.numpy().tolist().index(b'she')
        put_index = layer.token_strings.numpy().tolist().index(b'put')
        on_index = layer.token_strings.numpy().tolist().index(b'on')
        her_index = layer.token_strings.numpy().tolist().index(b'her')
        pajamas_index = layer.token_strings.numpy().tolist().index(b'pajamas')

        she_embedding = layer.token_embeddings.numpy().tolist()[she_index]
        layer(tf.constant(layer_path), tf.constant('she'), tf.constant(she_embedding), tf.constant(False), tf.constant(False))

        put_embedding = layer.token_embeddings.numpy().tolist()[put_index]
        layer(tf.constant(layer_path), tf.constant('put'), tf.constant(put_embedding), tf.constant(False), tf.constant(False))

        on_embedding = layer.token_embeddings.numpy().tolist()[on_index]
        layer(tf.constant(layer_path), tf.constant('on'), tf.constant(on_embedding), tf.constant(False), tf.constant(False))

        her_embedding = layer.token_embeddings.numpy().tolist()[her_index]
        layer(tf.constant(layer_path), tf.constant('her'), tf.constant(her_embedding), tf.constant(False), tf.constant(False))

        pajamas_embedding = layer.token_embeddings.numpy().tolist()[pajamas_index]
        layer(tf.constant(layer_path), tf.constant('pajamas'), tf.constant(pajamas_embedding), tf.constant(False), tf.constant(False))

        #assert she_index == 0
        assert layer.token_history[5, 0, start_of_line_index].numpy() == 1
        assert layer.token_history[4, 0, she_index].numpy() == 1
        assert layer.token_history[3, 0, put_index].numpy() == 1
        assert layer.token_history[2, 0, on_index].numpy() == 1
        assert layer.token_history[1, 0, her_index].numpy() == 1
        assert layer.token_history[0, 0, pajamas_index].numpy() == 1
