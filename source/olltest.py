import numpy as np
from ollama import Client

OLLAMA_URL = "http://192.168.1.142:11434"

def dot(va, vb):
    return sum(a * b for a, b in zip(va, vb))

def embed(text):
    client = Client(OLLAMA_URL)
    model_name = "nomic-embed-text:latest"
    
    # Generate embeddings using the Ollama client
    response = client.embed(model=model_name, input=text)
    
    # Return the embeddings
    return response.embeddings


def test_ollama():
    client = Client(OLLAMA_URL)
    model_name = "nomic-embed-text:latest"
    
    # Check if the model exists
    models = client.list().models
    model_names = [model.model for model in models]
    assert model_name in model_names, f"Model {model_name} not found in Ollama."

    # Generate a response from the model
    response = client.embed(model=model_name, input=["Hello, how are you?", "I am fine, thank you!"])
    
    # Check if the response is valid
    print(f'Response has {len(response.embeddings)} embeddings of length {len(response.embeddings[0])}.')
    #assert isinstance(response, str), "Response should be a string."
    #assert len(response) > 0, "Response should not be empty."
    
    print("Test passed successfully!")

if __name__ == "__main__":
    docs=["tell me about embeddings", "I am fine, thank you!"]
    docs_embed = embed(['query: ' + d for d in docs])
    print(f'Response has {len(docs_embed)} embeddings of length {len(docs_embed[0])}.')


    query = 'tell me about embeddings'
    query_embed = embed(['query: ' + query])[0]
    print(f'query {query}')

    for d, e in zip(docs, docs_embed):
        print(f'similarity {dot(query_embed, e):.2f}: {d!r}')