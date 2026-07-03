import os
import sys
import json

from ollama import Client
import tensorflow as tf
import numpy as np

OLLAMA_HOST = '192.168.1.142'
OLLAMA_PORT = 11434
OLLAMA_URL = f"http://{OLLAMA_HOST}:{OLLAMA_PORT}"
OLLAMA_MODEL = "embeddinggemma"

EMPTY_EMBEDDING = [0.0] * 768  # Assuming the embedding size is 768, adjust as necessary
EMPTY_EMBEDDING[0] = 1.0  # Set the first element to 1.0 to indicate an empty embedding

path = '/record/encoder/'

def Run(words_json):
  embeddings = []

  client = Client(OLLAMA_URL)

  for word in words_json:
    response = client.embed(OLLAMA_MODEL, word)
    embedding = response.embeddings[0] if response.embeddings else None
    if embedding is None:
      print(f"Failed to get embedding for word: {word}, using empty embedding.")
      embedding = EMPTY_EMBEDDING
    embeddings.append(embedding)

  similarity_matrix = [[]]
  for i in range(len(embeddings)):
    row = []
    for j in range(len(embeddings)):
      if i == j:
        row.append(1.0)  # Similarity with itself is always 1
      else:
        #sim = np.dot(embeddings[i], embeddings[j]) / (np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[j]))
        sim = tf.tensordot(tf.constant(embeddings[i]), tf.constant(embeddings[j]), axes=1)
        row.append(sim.numpy())  # Convert TensorFlow tensor to a NumPy scalar

    similarity_matrix.append(row)

  # Save the similarity matrix to a CSV file
  output_file = path + 'similarity_matrix' + sys.argv[1] + '.csv'
  with open(output_file, 'w') as f:
    for row in similarity_matrix:
      f.write(','.join(map(str, row)) + '\n')

   
#  Execution starts here.
if __name__ == "__main__":
  if len(sys.argv) < 2:
    print(f'Usage: {sys.argv[0]} <word file>')
    exit(0)

  wordfile = path + sys.argv[1]
  if not wordfile.endswith('.json'):
    wordfile += '.json'

  if not os.path.exists(wordfile):
    print(f'Word file {wordfile} does not exist')
    exit(0)

  with open(wordfile, 'r') as file:
    words_json = json.load(file)

  Run(words_json)
