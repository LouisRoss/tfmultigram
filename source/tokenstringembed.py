import sys
from ollama import Client
import tensorflow as tf

from settings import Settings
from tokenbase import TokenBase
from tfnodehelper import EmbeddingModule

OLLAMA_HOST = '192.168.1.142'
OLLAMA_PORT = 11434
OLLAMA_URL = f"http://{OLLAMA_HOST}:{OLLAMA_PORT}"
OLLAMA_MODEL = "embeddinggemma"

EMPTY_EMBEDDING = [0.0] * 768  # Assuming the embedding size is 768, adjust as necessary
EMPTY_EMBEDDING[0] = 1.0  # Set the first element to 1.0 to indicate an empty embedding

def dot(va, vb):
    return sum(a * b for a, b in zip(va, vb))

class TokenStringEmbed(TokenBase):
    """
    TokenStringEmbed is a concrete implementation of the TokenBase class.
    It represents a token that is a string and provides implementations
    for the abstract methods defined in TokenBase using ollama embeddings.
    This class assumes that an ollama server is running and accessible.
    """
    settings = Settings()
    string_register = [None for _ in range(settings.embeddings['embedding_count'])]
    threshold_score = settings.embeddings['threshold_score']
    embedding_register = EmbeddingModule(threshold_score, 'embedding_register')
    client = Client(OLLAMA_URL)


    def __init__(self, value: str):
        super().__init__('TokenStringEmbed')
        self.token_raw = value
        self.end_of_line = False

        response = TokenStringEmbed.client.embed(model=OLLAMA_MODEL, input=value)

        if len(response.embeddings) == 0:
            # If no embeddings are returned, use an empty embedding
            self.embedding = EMPTY_EMBEDDING
        else:
            self.embedding = response.embeddings[0]



    # Override methods from TokenBase
    def FindTokenIfSeen(self, tokens: list['TokenBase'], threshold_score: float = 1.0) -> 'TokenBase':
        """
        Examine all tokens for any that recognize this token.
        returns: The token already in the cache if it exists, or None if not found.
        """
        similarity, index = TokenStringEmbed.embedding_register('./', tf.constant(self.embedding, dtype=tf.float32))
        # print(f"Token {self.token_raw} Similarity: {similarity} to index {index}, Threshold Score: {threshold_score}")

        # If we were already seen, the similarity will be 1.0
        if similarity > TokenStringEmbed.threshold_score:
            return TokenStringEmbed.string_register[index]
        
        print(f"E[{index}]='{self.token_raw}'", end='  ')
        TokenStringEmbed.string_register[index] = self
        
        return None
    
    
    # Concrete implementation of abstract methods from TokenBase
    def CheckIfTokenSimilar(self, ref_token: TokenBase) -> int:
        """
        Return a measure of the similarity between this token and refToken.
        As a string, we choose a binary answer of 0 (not the same) or
        sys.maxsize (identical).
        ref_token: Reference token to compare with this token
        returns: Similarity measure between 0 and sys.maxsize
        """
        if not isinstance(ref_token,  TokenStringEmbed):
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
        if not isinstance(ref_token,  TokenStringEmbed):
            return False

        return self.CheckIfTokenSimilar(ref_token) > TokenStringEmbed.threshold_score    
    def GetAsString(self) -> str:
        """
        For logging and analysis, get this token as a string.
        For string type, this is easy.
        returns: The string value this token encodes.
        """
        if self.end_of_line:
            return "<eol>"
        
        return self.token_raw
    
