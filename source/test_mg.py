import pytest

from multigramconfiguration import MultigramConfiguration
from base_initializer import BaseInitializer
from tfprogram import LayerModule
import tensorflow as tf


@pytest.fixture
def setup_layer():

    config = MultigramConfiguration('sequence_1')
    init = BaseInitializer(config)
    layer = LayerModule(config, init)

    layer.connections[0].assign([[0, 2, 0, 0],[1, 0, 0, 0],[1, 0, 0, 0],[0, 0, 4, 0]])
    layer.connections[1].assign([[0, 0, 3, 0],[0, 0, 0, 0],[0, 0, 1, 0],[0, 1, 0, 0]])
    layer.connections[2].assign([[2, 0, 0, 0],[1, 0, 0, 0],[1, 0, 0, 0],[1, 0, 0, 0]])
    layer.connections[3].assign([[2, 0, 0, 0],[1, 0, 0, 0],[1, 0, 0, 0],[1, 0, 0, 0]])

    layer.token_history[0].assign([0,1,1,0])
    yield layer

input_tokens = [[1],[0],[0],[0], 
                [0],[1],[0],[0], 
                [0],[0],[1],[0], 
                [1],[0],[0],[1]]

class TestMultigramLayer:
    def test_multigram_forward_connect(self, setup_layer):
        layer = setup_layer

        # Add test tokens to input.
        layer.tokens.assign([[1],[0],[0],[1]])

        layer.ForwardConnectTokens()
        assert layer.activeconnections.shape == (8, 4, 4)
        assert all((layer.activeconnections[i].numpy().flatten() == [1,1,1,1, 0,0,0,0, 0,0,0,0, 1,1,1,1]).all() for i in range(8))

    def test_multigram_connect_history(self, setup_layer):
        layer = setup_layer

        # Add test tokens to input.
        layer.tokens.assign([[1],[0],[0],[1]])

        layer.ForwardConnectTokens()
        layer.ConnectHistory()
        assert layer.connections.shape == (8, 4, 4)
        assert layer.connections[0].numpy().flatten().tolist() == [0,3,1,0, 1,0,0,0, 1,0,0,0, 0,1,5,0]
        assert layer.connections[3].numpy().flatten().tolist() == [2,0,0,0, 1,0,0,0, 1,0,0,0, 1,0,0,0]

    def test_multigram_predict_next_token(self, setup_layer):
        layer = setup_layer

        # Add test tokens to input.
        layer.tokens.assign([[1],[0],[0],[1]])

        layer.ForwardConnectTokens()
        layer.ConnectHistory()
        layer.PredictNextToken()
        assert tf.reduce_sum(layer.activeconnections * layer.connections, axis=0).numpy().flatten().tolist() == [4,3,4,0, 0,0,0,0, 0,0,0,0, 2,2,5,0]
        assert layer.token_predictions.shape == (4,)
        assert layer.token_predictions.numpy().tolist() == [6,5,9,0]

