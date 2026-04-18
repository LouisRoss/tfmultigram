from multigramconfiguration import MultigramConfiguration
from base_initializer import BaseInitializer
from tfprogram import LayerModule
import tensorflow as tf


config = MultigramConfiguration('sequence_1')
init = BaseInitializer(config)
layer = LayerModule(config, init)

# Add test tokens to input.
layer.tokens.assign([[1],[0],[0],[1]])

layer.connections[0].assign([[0, 2, 0, 0],[1, 0, 0, 0],[1, 0, 0, 0],[0, 0, 4, 0]])
layer.connections[1].assign([[0, 0, 3, 0],[0, 0, 0, 0],[0, 0, 1, 0],[0, 1, 0, 0]])
layer.connections[2].assign([[2, 0, 0, 0],[1, 0, 0, 0],[1, 0, 0, 0],[1, 0, 0, 0]])
layer.connections[3].assign([[2, 0, 0, 0],[1, 0, 0, 0],[1, 0, 0, 0],[1, 0, 0, 0]])

layer.token_history[0].assign([0,1,1,0])
