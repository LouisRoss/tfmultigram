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
        assert layer.token_activations.shape == (configuration.GetMaxDistance(), configuration.GetLayerSize(), 1)
        assert layer.activeconnections.shape == (configuration.GetMaxDistance(), configuration.GetLayerSize(), configuration.GetLayerSize())

    @pytest.mark.filterwarnings("ignore:DeprecationWarning")
    def test_connect_five_words(self, load_embeddings, setup_layer):
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

    @pytest.mark.filterwarnings("ignore:DeprecationWarning")
    def test_forward_start(self, load_embeddings, setup_layer):
        embeddings = load_embeddings
        layer = setup_layer

        # Model loads from disk on first call.
        start_of_line_embedding = Get_global_embedding('>>>')
        layer(tf.constant(layer_path), tf.constant('>>>'), tf.constant(start_of_line_embedding), tf.constant(False), tf.constant(False))

        start_of_line_index = layer.token_strings.numpy().tolist().index(b'>>>')
        assert layer.tokens[start_of_line_index, 0].numpy() == 1, "The first token should be activated."
        assert layer.token_activations[0, start_of_line_index, 0].numpy() == 1, "The first token should be activated."
        assert layer.activeconnections[0, start_of_line_index, 0].numpy() == 1, "The first token should be activated at all distances."
        assert layer.activeconnections[1, start_of_line_index, 0].numpy() == 1, "The first token should be activated at all distances."
        assert layer.activeconnections[configuration.GetMaxDistance()-1, start_of_line_index, 0].numpy() == 1, "The first token should be activated at all distances."

    @pytest.mark.filterwarnings("ignore:DeprecationWarning")
    def test_forward_first_word(self, load_embeddings, setup_layer):
        embeddings = load_embeddings
        layer = setup_layer

        # Model loads from disk on first call.
        start_of_line_embedding = Get_global_embedding('>>>')
        layer(tf.constant(layer_path), tf.constant('>>>'), tf.constant(start_of_line_embedding), tf.constant(False), tf.constant(False))

        she_index = layer.token_strings.numpy().tolist().index(b'she')
        tf.print(f'she_index: {she_index}')
        she_embedding = layer.token_embeddings.numpy().tolist()[she_index]
        layer(tf.constant(layer_path), tf.constant('she'), tf.constant(she_embedding), tf.constant(False), tf.constant(False))

        assert layer.tokens[she_index, 0].numpy() == 1, "The first token should be activated."
        assert layer.token_activations[0, she_index, 0].numpy() == 1, "The first word token should be activated."
        assert layer.activeconnections[0, she_index, 0].numpy() == 1, "The first word token should be activated at all distances."
        assert layer.activeconnections[1, she_index, 0].numpy() == 1, "The first word token should be activated at all distances."
        assert layer.activeconnections[configuration.GetMaxDistance()-1, she_index, 0].numpy() == 1, "The first word token should be activated at all distances."
        assert layer.activeconnections[0, she_index, 1].numpy() == 1, "The first word token should be activated."
        assert layer.activeconnections[0, she_index, configuration.GetMaxDistance()-1].numpy() == 1, "The first word token should be activated."
        assert layer.activeconnections[configuration.GetMaxDistance()-1, she_index, configuration.GetMaxDistance()-1].numpy() == 1, "The first word token should be activated."

    @pytest.mark.filterwarnings("ignore:DeprecationWarning")
    def test_activate_two_words(self, load_embeddings, setup_layer):
        embeddings = load_embeddings
        layer = setup_layer

        # Send start of line token.
        start_of_line_embedding = Get_global_embedding('>>>')
        # Model loads from disk on first call.
        layer(tf.constant(layer_path), tf.constant('>>>'), tf.constant(start_of_line_embedding), tf.constant(False), tf.constant(False))
        start_of_line_index = layer.token_strings.numpy().tolist().index(b'>>>')

        # Send 'she' token.
        she_index = layer.token_strings.numpy().tolist().index(b'she')
        tf.print(f'she_index: {she_index}')
        she_embedding = layer.token_embeddings.numpy().tolist()[she_index]
        old_start_she_connection = layer.connections[0, she_index, start_of_line_index].numpy()
        layer(tf.constant(layer_path), tf.constant('she'), tf.constant(she_embedding), tf.constant(False), tf.constant(False))
        # Check that layer.connectedhistory and layer.connections are updated correctly after each token is processed.
        assert layer.connectedhistory[0, she_index, start_of_line_index].numpy() == 1, "Distance-1 activation from 'start_of_line' to 'she' should be detected."
        assert layer.connections[0, she_index, start_of_line_index].numpy() == old_start_she_connection + 1, "Distance-1 connection from 'start_of_line' to 'she' should be bumped."

        # Send 'put' token.
        put_index = layer.token_strings.numpy().tolist().index(b'put')
        tf.print(f'put_index: {put_index}')
        put_embedding = layer.token_embeddings.numpy().tolist()[put_index]
        old_she_put_connection = layer.connections[0, put_index, she_index].numpy()
        old_start_put_connection = layer.connections[1, put_index, start_of_line_index].numpy()
        layer(tf.constant(layer_path), tf.constant('put'), tf.constant(put_embedding), tf.constant(False), tf.constant(False))
        # Check that layer.connectedhistory and layer.connections are updated correctly after each token is processed.
        assert layer.connectedhistory[0, put_index, she_index].numpy() == 1, "Distance-1 activation from 'she' to 'put' should be detected."
        assert layer.connections[0, put_index, she_index].numpy() == old_she_put_connection + 1, "Distance-1 connection from 'she' to 'put' should be bumped."
        assert layer.connectedhistory[1, put_index, start_of_line_index].numpy() == 1, "Distance-2 activation from 'start_of_line' to 'put' should be detected."
        assert layer.connections[1, put_index, start_of_line_index].numpy() == old_start_put_connection + 1, "Distance-2 connection from 'start_of_line' to 'put' should be bumped."


    @pytest.mark.filterwarnings("ignore:DeprecationWarning")
    def test_activate_five_words(self, load_embeddings, setup_layer):
        embeddings = load_embeddings
        layer = setup_layer

        # Send start of line token.
        start_of_line_embedding = Get_global_embedding('>>>')
        # Model loads from disk on first call.
        layer(tf.constant(layer_path), tf.constant('>>>'), tf.constant(start_of_line_embedding), tf.constant(False), tf.constant(False))
        start_of_line_index = layer.token_strings.numpy().tolist().index(b'>>>')

        # Send 'she' token.
        she_index = layer.token_strings.numpy().tolist().index(b'she')
        tf.print(f'she_index: {she_index}')
        she_embedding = layer.token_embeddings.numpy().tolist()[she_index]
        old_start_she_connection = layer.connections[0, she_index, start_of_line_index].numpy()
        layer(tf.constant(layer_path), tf.constant('she'), tf.constant(she_embedding), tf.constant(False), tf.constant(False))
        assert layer.connectedhistory[0, she_index, start_of_line_index].numpy() == 1, "Distance-1 activation from 'start_of_line' to 'she' should be detected."
        assert layer.connections[0, she_index, start_of_line_index].numpy() == old_start_she_connection + 1, "Distance-1 connection from 'start_of_line' to 'she' should be bumped."

        # Send 'put' token.
        put_index = layer.token_strings.numpy().tolist().index(b'put')
        tf.print(f'put_index: {put_index}')
        put_embedding = layer.token_embeddings.numpy().tolist()[put_index]
        old_she_put_connection = layer.connections[0, put_index, she_index].numpy()
        old_start_put_connection = layer.connections[1, put_index, start_of_line_index].numpy()
        layer(tf.constant(layer_path), tf.constant('put'), tf.constant(put_embedding), tf.constant(False), tf.constant(False))
        assert layer.connectedhistory[0, put_index, she_index].numpy() == 1, "Distance-1 activation from 'she' to 'put' should be detected."
        assert layer.connections[0, put_index, she_index].numpy() == old_she_put_connection + 1, "Distance-1 connection from 'she' to 'put' should be bumped."
        assert layer.connectedhistory[1, put_index, start_of_line_index].numpy() == 1, "Distance-2 activation from 'start_of_line' to 'put' should be detected."
        assert layer.connections[1, put_index, start_of_line_index].numpy() == old_start_put_connection + 1, "Distance-2 connection from 'start_of_line' to 'put' should be bumped."

        # Send 'on' token.
        on_index = layer.token_strings.numpy().tolist().index(b'on')
        tf.print(f'on_index: {on_index}')
        on_embedding = layer.token_embeddings.numpy().tolist()[on_index]
        old_put_on_connection = layer.connections[0, on_index, put_index].numpy()
        old_she_on_connection = layer.connections[1, on_index, she_index].numpy()
        old_start_on_connection = layer.connections[2, on_index, start_of_line_index].numpy()
        layer(tf.constant(layer_path), tf.constant('on'), tf.constant(on_embedding), tf.constant(False), tf.constant(False))
        assert layer.connectedhistory[0, on_index, put_index].numpy() == 1, "Distance-1 activation from 'put' to 'on' should be detected."
        assert layer.connections[0, on_index, put_index].numpy() == old_put_on_connection + 1, "Distance-1 connection from 'put' to 'on' should be bumped."
        assert layer.connectedhistory[1, on_index, she_index].numpy() == 1, "Distance-2 activation from 'she' to 'on' should be detected."
        assert layer.connections[1, on_index, she_index].numpy() == old_she_on_connection + 1, "Distance-2 connection from 'she' to 'on' should be bumped."
        assert layer.connectedhistory[2, on_index, start_of_line_index].numpy() == 1, "Distance-3 activation from 'start_of_line' to 'on' should be detected."
        assert layer.connections[2, on_index, start_of_line_index].numpy() == old_start_on_connection + 1, "Distance-3 connection from 'start_of_line' to 'on' should be bumped."

        # Send 'her' token.
        her_index = layer.token_strings.numpy().tolist().index(b'her')
        tf.print(f'her_index: {her_index}')
        her_embedding = layer.token_embeddings.numpy().tolist()[her_index]
        old_on_her_connection = layer.connections[0, her_index, on_index].numpy()
        old_put_her_connection = layer.connections[1, her_index, put_index].numpy()
        old_she_her_connection = layer.connections[2, her_index, she_index].numpy()
        old_start_her_connection = layer.connections[3, her_index, start_of_line_index].numpy()
        layer(tf.constant(layer_path), tf.constant('her'), tf.constant(her_embedding), tf.constant(False), tf.constant(False))
        assert layer.connectedhistory[0, her_index, on_index].numpy() == 1, "Distance-1 activation from 'on' to 'her' should be detected."
        assert layer.connections[0, her_index, on_index].numpy() == old_on_her_connection + 1, "Distance-1 connection from 'on' to 'her' should be bumped."
        assert layer.connectedhistory[1, her_index, put_index].numpy() == 1, "Distance-2 activation from 'put' to 'her' should be detected."
        assert layer.connections[1, her_index, put_index].numpy() == old_put_her_connection + 1, "Distance-2 connection from 'put' to 'her' should be bumped."
        assert layer.connectedhistory[2, her_index, she_index].numpy() == 1, "Distance-3 activation from 'she' to 'her' should be detected."
        assert layer.connections[2, her_index, she_index].numpy() == old_she_her_connection + 1, "Distance-3 connection from 'she' to 'her' should be bumped."
        assert layer.connectedhistory[3, her_index, start_of_line_index].numpy() == 1, "Distance-4 activation from 'start_of_line' to 'on' should be detected."
        assert layer.connections[3, her_index, start_of_line_index].numpy() == old_start_her_connection + 1, "Distance-4 connection from 'start_of_line' to 'on' should be bumped."

        # Send 'pajamas' token.
        pajamas_index = layer.token_strings.numpy().tolist().index(b'pajamas')
        tf.print(f'pajamas_index: {pajamas_index}')
        pajamas_embedding = layer.token_embeddings.numpy().tolist()[pajamas_index]
        old_her_pajamas_connection = layer.connections[0, pajamas_index, her_index].numpy()
        old_on_pajamas_connection = layer.connections[1, pajamas_index, on_index].numpy()
        old_put_pajamas_connection = layer.connections[2, pajamas_index, put_index].numpy()
        old_she_pajamas_connection = layer.connections[3, pajamas_index, she_index].numpy()
        old_start_pajamas_connection = layer.connections[4, pajamas_index, start_of_line_index].numpy()
        layer(tf.constant(layer_path), tf.constant('pajamas'), tf.constant(pajamas_embedding), tf.constant(False), tf.constant(False))
        assert layer.connectedhistory[0, pajamas_index, her_index].numpy() == 1, "Distance-1 activation from 'her' to 'pajamas' should be detected."
        assert layer.connections[0, pajamas_index, her_index].numpy() == old_her_pajamas_connection + 1, "Distance-1 connection from 'her' to 'pajamas' should be bumped."
        assert layer.connectedhistory[1, pajamas_index, on_index].numpy() == 1, "Distance-2 activation from 'on' to 'pajamas' should be detected."
        assert layer.connections[1, pajamas_index, on_index].numpy() == old_on_pajamas_connection + 1, "Distance-2 connection from 'on' to 'pajamas' should be bumped."
        assert layer.connectedhistory[2, pajamas_index, put_index].numpy() == 1, "Distance-3 activation from 'put' to 'pajamas' should be detected."
        assert layer.connections[2, pajamas_index, put_index].numpy() == old_put_pajamas_connection + 1, "Distance-3 connection from 'put' to 'pajamas' should be bumped."
        assert layer.connectedhistory[3, pajamas_index, she_index].numpy() == 1, "Distance-4 activation from 'she' to 'pajamas' should be detected."
        assert layer.connections[3, pajamas_index, she_index].numpy() == old_she_pajamas_connection + 1, "Distance-4 connection from 'she' to 'pajamas' should be bumped."
        assert layer.connectedhistory[4, pajamas_index, start_of_line_index].numpy() == 1, "Distance-5 activation from 'start_of_line' to 'pajamas' should be detected."
        assert layer.connections[4, pajamas_index, start_of_line_index].numpy() == old_start_pajamas_connection + 1, "Distance-5 connection from 'start_of_line' to 'pajamas' should be bumped."

    @pytest.mark.filterwarnings("ignore:DeprecationWarning")
    def test_fire_first_word(self, load_embeddings, setup_layer):
        embeddings = load_embeddings
        layer = setup_layer

        start_of_line_embedding = Get_global_embedding('>>>')
        # Model loads from disk on first call.
        layer(tf.constant(layer_path), tf.constant('>>>'), tf.constant(start_of_line_embedding), tf.constant(False), tf.constant(False))
        start_of_line_index = layer.token_strings.numpy().tolist().index(b'>>>')
        assert layer.token_firing_history[0, 0, start_of_line_index].numpy() == layer.maxdistance - 1, "The first word token should leave firing history."

        she_index = layer.token_strings.numpy().tolist().index(b'she')
        put_index = layer.token_strings.numpy().tolist().index(b'put')
        tf.print(f'she_index: {she_index}')
        she_embedding = layer.token_embeddings.numpy().tolist()[she_index]
        layer(tf.constant(layer_path), tf.constant('she'), tf.constant(she_embedding), tf.constant(False), tf.constant(False))
        assert layer.token_firing_history[0, 0, she_index].numpy() == layer.maxdistance - 1, "The second word token should leave firing history."
        assert layer.token_firing_history[1, 0, start_of_line_index].numpy() == layer.maxdistance - 2, "The first word token should leave firing history."
        expanded_firing_history = tf.broadcast_to(layer.token_firing_history, [layer.maxdistance, layer.layer_size, layer.layer_size])
        assert expanded_firing_history[0, 0, she_index].numpy() == layer.maxdistance - 1, "The second word token should leave firing history."
        assert expanded_firing_history[0, 1, she_index].numpy() == layer.maxdistance - 1, "The second word token should leave firing history."
        assert expanded_firing_history[0, layer.maxdistance - 1, she_index].numpy() == layer.maxdistance - 1, "The second word token should leave firing history."

        # Repeat some internal calculations to verify the synaptic contribution is computed correctly
        weighted_firing_history = expanded_firing_history * layer.connections

        # Check that the connections are set up correctly
        put_index = layer.token_strings.numpy().tolist().index(b'put')
        on_index = layer.token_strings.numpy().tolist().index(b'on')
        assert weighted_firing_history[0, put_index, she_index].numpy() == (layer.maxdistance - 1) * 8, "Connection from 'they' to 'also' should be set."
        assert weighted_firing_history[1, on_index, start_of_line_index].numpy() == (layer.maxdistance - 2) * 2, "Connection from 'they' to 'helped' should be set."

        synaptic_contribution = tf.reduce_sum(weighted_firing_history, axis=0)
        tf.print(f'She index: {she_index}, Put index: {put_index}, On index: {on_index}, Start of line index: {start_of_line_index}')
        tf.print(f'synaptic_contribution: {synaptic_contribution[put_index].numpy()}')
        assert synaptic_contribution[put_index, she_index].numpy() == (layer.maxdistance - 1) * 8, "Connection from 'they' to 'also' should be set."
        assert synaptic_contribution[on_index, start_of_line_index].numpy() == (layer.maxdistance - 2) * 2, "Connection from 'they' to 'helped' should be set."

        assert layer.token_predictions[she_index].numpy() == 270, "Connection from 'they' to 'also' should be set."
        assert layer.token_predictions[put_index].numpy() == 188, "Connection from 'they' to 'also' should be set."
        assert layer.token_predictions[start_of_line_index].numpy() == 0, "Connection from 'they' to 'helped' should be set."
