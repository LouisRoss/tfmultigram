from ollama import Client
import tensorflow as tf

from settings import Settings
from tfnodehelper import EmbeddingModule

OLLAMA_HOST = '192.168.1.142'
OLLAMA_PORT = 11434
OLLAMA_URL = f"http://{OLLAMA_HOST}:{OLLAMA_PORT}"
OLLAMA_MODEL = "embeddinggemma"

EMPTY_EMBEDDING = [0.0] * 768  # Assuming the embedding size is 768, adjust as necessary
EMPTY_EMBEDDING[0] = 1.0  # Set the first element to 1.0 to indicate an empty embedding

def dot(va, vb):
    return sum(a * b for a, b in zip(va, vb))

class TfTokenString:
    """
    TfTokenString represents a token that is a string and uses ollama embeddings.
    This class assumes that an ollama server is running and accessible.
    """
    settings = Settings()
    string_register = [None for _ in range(settings.embeddings['embedding_count'])]
    threshold_score = settings.embeddings['threshold_score']
    embedding_register = EmbeddingModule(threshold_score, 'embedding_register')
    client = Client(OLLAMA_URL)


    def __init__(self, value: str):
        self.token_raw = value
        self.end_of_line = False

        response = TfTokenString.client.embed(model=OLLAMA_MODEL, input=value)

        if len(response.embeddings) == 0:
            # If no embeddings are returned, use an empty embedding
            self.embedding = EMPTY_EMBEDDING
        else:
            self.embedding = response.embeddings[0]



    def FindTokenIfSeen(self, tokens: list['TfTokenString'], threshold_score: float = 1.0) -> 'TfTokenString':
        """
        Examine all tokens for any that recognize this token.
        returns: The token already in the cache if it exists, or None if not found.
        """
        similarity, index = TfTokenString.embedding_register('./', tf.constant(self.embedding, dtype=tf.float32))
        # print(f"Token {self.token_raw} Similarity: {similarity} to index {index}, Threshold Score: {threshold_score}")

        # If we were already seen, the similarity will be 1.0
        if similarity > TfTokenString.threshold_score:
            return TfTokenString.string_register[index]
        
        print(f"E[{index}]='{self.token_raw}'", end='  ')
        TfTokenString.string_register[index] = self
        
        return None
    
    
    def CheckIfTokenSimilar(self, ref_token: 'TfTokenString') -> int:
        """
        Return a measure of the similarity between this token and ref_token.
        ref_token: Reference token to compare with this token
        returns: Similarity measure between 0 and sys.maxsize
        """
        if not isinstance(ref_token,  TfTokenString):
            return 0

        return dot(ref_token.embedding, self.embedding)
    
    def IsEqualTo(self, ref_token):
        """
        Boolean equality.  True if this token and refToken encode the same value.
        For strings, this gives the same results as CheckIfTokenSimilar,
        but other types may work differently.
        /// </summary>
        ref_token: A reference token to compare with this token
        returns: True if tokens encode the same value, false otherwise
        """
        if not isinstance(ref_token,  TfTokenString):
            return False

        return self.CheckIfTokenSimilar(ref_token) > TfTokenString.threshold_score
    
    def GetAsString(self) -> str:
        """
        For logging and analysis, get this token as a string.
        For string type, this is easy.
        returns: The string value this token encodes.
        """
        if self.end_of_line:
            return "<eol>"
        
        return self.token_raw
    
