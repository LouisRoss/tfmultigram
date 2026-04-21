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
    def test_multigram_layer(self, setup_layer):
        layer = setup_layer

        # Add test tokens to input.
        layer.tokens.assign([[1],[0],[0],[1]])

        layer.PropagateTokens()
        assert layer.token_timers.shape == (8, 4, 1) 

