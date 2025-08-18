import re
import os
import tensorflow as tf
import numpy as np
from settings import Settings

class EmbeddingModule(tf.Module):
  """
  This class extends the Tensorflow Module class, so that any method decorated
  with the @tf.function notation will be compiled into a compute graph, on first
  execution, and all subsequent iterations will run on the compute device.
  The functor of this class implements a single tick of the spiking neural algorithm,
  including learning.
  """

  def __init__(self, threshold_score, name=None):
    super().__init__(name=name)
    self.settings = Settings()
    self.threshold_score = threshold_score

    self.is_built = False


  def add_embedding(self, embedding):
    """
    Adds a new embedding to the module.
    """
    added_index = -1

    similarity, index = self.FindSimilarEmbedding(embedding)
    if similarity > 0.99:
      # If the embedding is similar to an existing one, do not add it again.
      return similarity, added_index
    
    if self.current_embedding < self.settings.embeddings['embedding_count']:
      self.string_embeddings[self.current_embedding].assign(embedding)
      added_index = self.current_embedding
      self.current_embedding += 1
    else:
      raise ValueError(f"Maximum number of embeddings {self.settings.embeddings['embedding_count']} reached.")
    
    return similarity, added_index

  def FindSimilarEmbedding(self, ref_embedding):
    """
    """
    # Do the dot product, broadcasting the reference embedding by duplicating it to match the shape of string_embeddings.
    ewp = tf.multiply(ref_embedding, self.string_embeddings)
    dot_product = tf.reduce_sum(ewp, axis=1)

    # Find the index of the maximum dot product.
    index = tf.argmax(dot_product)

    # Return the similarity and index of the most similar embedding.
    return dot_product[index], index


  @tf.function
  def __call__(self, datafolder, ref_embedding, log=False):
    # Create variables on first call.
    if not self.is_built:
      print(f"Initializing embeddings with max_embedding: {self.settings.embeddings['embedding_count']}")
      self.max_embedding = tf.Variable(self.settings.embeddings['embedding_count'], dtype=tf.int64)
      self.embedding_length = tf.constant(self.settings.embeddings['embedding_length'])
      self.string_embeddings = tf.Variable(tf.zeros([self.max_embedding, self.embedding_length], dtype=tf.float32), name='string_embeddings', trainable=False)
      self.current_embedding = tf.Variable(0, dtype=tf.int64)
      self.current_index = tf.Variable(0, dtype=tf.int64)
      self.threshold = tf.Variable(self.threshold_score, dtype=tf.float32)

      self.is_built = True

    similarity, index = self.FindSimilarEmbedding(ref_embedding)
    self.current_index.assign(index)
    if tf.less(similarity, self.threshold):
      if tf.less(self.current_embedding, self.max_embedding):
        self.string_embeddings[self.current_embedding].assign(ref_embedding)
        self.current_index.assign(self.current_embedding)
        self.current_embedding.assign_add(1)


    if log:
      pass
      #tf.print(self.connections, summarize=-1, sep=',', output_stream= 'file://' + datafolder + 'fullconnections.dat')
      #tf.print(self.spikes, summarize=-1, sep=',', output_stream= 'file://' + datafolder + 'fullspike.dat')
      #tf.print(self.potentials, summarize=-1, sep=',', output_stream= 'file://' + datafolder + 'fullactivations.dat')
      #tf.print(self.hebbtimers, summarize=-1, sep=',', output_stream= 'file://' + datafolder + 'fullhebbtimers.dat')

    return similarity, self.current_index
  
