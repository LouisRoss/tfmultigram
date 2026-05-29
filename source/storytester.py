import json

from tokensourcedataset import TokenSourceDataset
from ollama import Client

OLLAMA_HOST = '192.168.1.142'
OLLAMA_PORT = 11434
OLLAMA_URL = f"http://{OLLAMA_HOST}:{OLLAMA_PORT}"
OLLAMA_MODEL = "embeddinggemma"

embedding_map = {}

def read_story_tokens(token_source, client):
    while token_source.IsInputAvailable():
      token = token_source.GetNext()

      print(f'{token.token_raw}', end=' ')
      response = client.embed(model=OLLAMA_MODEL, input=token.token_raw)
      embedding_map[token.token_raw] = response.embeddings[0]

      if token.end_of_line:
        print()


client = Client(OLLAMA_URL)

with TokenSourceDataset("roneneldan/TinyStories", 25) as token_source:
  read_story_tokens(token_source, client)

# Save dictionary to a file
with open("/record/embeddings.json", "w") as f:
    json.dump(embedding_map, f, indent=4)  # indent adds readability
