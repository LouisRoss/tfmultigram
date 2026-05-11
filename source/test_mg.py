import pytest

from multigramconfiguration import MultigramConfiguration
from base_initializer import BaseInitializer
from tfprogram import LayerModule
import tensorflow as tf

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


@pytest.fixture
def setup_layer():

    config = MultigramConfiguration('', test_config)
    init = BaseInitializer(config)
    layer = LayerModule(config, init)

    layer.connections[0].assign([[0, 2, 0, 0],[1, 0, 0, 0],[1, 0, 0, 0],[0, 0, 4, 0]])
    layer.connections[1].assign([[0, 0, 3, 0],[0, 0, 0, 0],[0, 0, 1, 0],[0, 1, 0, 0]])
    layer.connections[2].assign([[2, 0, 0, 0],[1, 0, 0, 0],[1, 0, 0, 0],[1, 0, 0, 0]])
    layer.connections[3].assign([[2, 0, 0, 0],[1, 0, 0, 0],[1, 0, 0, 0],[1, 0, 0, 0]])

    layer.token_history[0].assign([0,1,1,0])
    yield layer

@pytest.fixture
def setup_empty_layer():

    config = MultigramConfiguration('', test_config)
    init = BaseInitializer(config)
    layer = LayerModule(config, init)

    yield layer

input_tokens = [[1],[0],[0],[0], 
                [0],[1],[0],[0], 
                [0],[0],[1],[0], 
                [1],[0],[0],[1]]

class TestMultigramLayer:
    def test_accept_token(self, setup_layer):
        layer = setup_layer

        # Add test token to input.
        layer.AcceptToken("test_token")

        assert layer.tokens.shape == (4, 1)
        assert tf.reduce_sum(layer.tokens).numpy() == 1
        assert layer.token_strings.shape == (4,)
        assert b'test_token' in layer.token_strings.numpy().tolist()
        assert layer.token_embeddings.shape == (4, 768)
        assert tf.reduce_sum(layer.token_embeddings[0]).numpy() != 0
        assert tf.reduce_sum(layer.token_embeddings[1]).numpy() == 0
        assert tf.reduce_sum(layer.token_embeddings[2]).numpy() == 0
        assert tf.reduce_sum(layer.token_embeddings[3]).numpy() == 0
        assert layer.current_new_token_index.numpy() == 1

    def test_accept_same_token(self, setup_layer):
        layer = setup_layer

        # Add test token to input.
        layer.AcceptToken("test_token")
        layer.AcceptToken("test_token")

        assert layer.tokens.shape == (4, 1)
        assert tf.reduce_sum(layer.tokens).numpy() == 1
        assert layer.token_strings.shape == (4,)
        assert b'test_token' in layer.token_strings.numpy().tolist()
        assert layer.token_embeddings.shape == (4, 768)
        assert tf.reduce_sum(layer.token_embeddings[0]).numpy() != 0
        assert tf.reduce_sum(layer.token_embeddings[1]).numpy() == 0
        assert tf.reduce_sum(layer.token_embeddings[2]).numpy() == 0
        assert tf.reduce_sum(layer.token_embeddings[3]).numpy() == 0
        assert layer.current_new_token_index.numpy() == 1

    def test_accept_multiple_tokens(self, setup_layer):
        layer = setup_layer

        # Add test token to input.
        layer.AcceptToken("test_token")
        layer.AcceptToken("forest_gump")

        assert layer.tokens.shape == (4, 1)
        assert tf.reduce_sum(layer.tokens).numpy() == 1
        assert layer.token_strings.shape == (4,)
        assert b'test_token' in layer.token_strings.numpy().tolist()
        assert b'forest_gump' in layer.token_strings.numpy().tolist()
        assert layer.token_embeddings.shape == (4, 768)
        assert tf.reduce_sum(layer.token_embeddings[0]).numpy() != 0
        assert tf.reduce_sum(layer.token_embeddings[1]).numpy() != 0
        assert tf.reduce_sum(layer.token_embeddings[2]).numpy() == 0
        assert tf.reduce_sum(layer.token_embeddings[3]).numpy() == 0
        assert layer.current_new_token_index.numpy() == 2

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

    def test_full_cycle(self, setup_empty_layer):
        layer = setup_empty_layer

        test_tokens = ["once", "upon", "time", "upon", "time"]

        # Add test tokens to input.
        for token in test_tokens:
            layer.AcceptToken(token)
            layer.ForwardConnectTokens()
            layer.ConnectHistory()
            layer.PredictNextToken()
            layer.PushTokenHistory()

        assert layer.tokens.numpy().flatten().tolist() == [0,0,1,0]
        assert layer.token_history[0].numpy().flatten().tolist() == [0,0,1,0]
        assert layer.token_history[1].numpy().flatten().tolist() == [0,1,0,0]
        assert layer.token_history[2].numpy().flatten().tolist() == [0,0,1,0]
        assert layer.token_history[3].numpy().flatten().tolist() == [0,1,0,0]
        assert layer.token_history[4].numpy().flatten().tolist() == [1,0,0,0]

        # Transitions at distance 1.
        # Token 0 -> 1 once.
        assert layer.connections[0, 1, 0].numpy() == 1
        # Token 1 -> 2 twice.
        assert layer.connections[0, 2, 1].numpy() == 2
        # Token 2 -> 1 once.
        assert layer.connections[0, 1, 2].numpy() == 1

        # Put it all together.
        assert layer.connections[0].numpy().flatten().tolist() == [0,0,0,0, 1,0,1,0, 0,2,0,0, 0,0,0,0]


        # Transitions at distance 2.
        # Token 0 -> 2 once.
        assert layer.connections[1, 2, 0].numpy() == 1
        # Token 1 -> 1 once.
        assert layer.connections[1, 1, 1].numpy() == 1

        # Put it all together.
        assert layer.connections[1].numpy().flatten().tolist() == [0,0,0,0, 0,1,0,0, 1,0,1,0, 0,0,0,0]


        # Transitions at distance 3.
        # Token 0 -> 1 once.
        assert layer.connections[2, 1, 0].numpy() == 1
        # Token 1 -> 2 once.
        assert layer.connections[2, 2, 1].numpy() == 1

        # Put it all together.
        assert layer.connections[2].numpy().flatten().tolist() == [0,0,0,0, 1,0,0,0, 0,1,0,0, 0,0,0,0]


        # Transitions at distance 4.
        # Token 0 -> 2 once.
        assert layer.connections[3, 2, 0].numpy() == 1

        # Put it all together.
        assert layer.connections[3].numpy().flatten().tolist() == [0,0,0,0, 0,0,0,0, 1,0,0,0, 0,0,0,0]
