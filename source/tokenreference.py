import sys
from tokenbase import TokenBase

class TokenReference(TokenBase):
    """
    TokenReference is a concrete implementation of the TokenBase class.
    It represents a token that is a reference to another token and provides
    specific implementations for the abstract methods defined in TokenBase.
    """

    def __init__(self, ref_tokens: list[TokenBase]):
        super().__init__()
        self.token_raw = ref_tokens

        # Tokens with fewer non-null bytes than this are not significant.
        # Token comparisons that are not true for at least this number of bytes are ignored.
        self.token_significant_size = 0
        self.unexpected = False

    def CheckIfTokenSimilar(self, ref_token: TokenBase) -> int:
        """
        Return a measure of the similarity between this token and refToken.
        As a reference-type token, similarity means the number of consecutive
        identical bytes before a nonidentical byte.  Scale similarity between 
        0 (different) and int.MaxValue (identical).
        ref_token: A reference token to check for similarity
        returns: Similarity measure between 0 and int.MaxValue
        """

        # Tokens are not the same if they are of different types.
        if not isinstance(ref_token, TokenReference):
            return 0
        
        size_difference = abs(len(self.token_raw) - len(ref_token.token_raw))

        # Count the number of consecutive identical bytes before the first nonidentical one.
        score = 0
        min_size = min(len(self.token_raw), len(ref_token.token_raw))
        for i in range(min_size):
            if self.token_raw[i].IsEqualTo(ref_token.token_raw[i]):
                score += 1

        score -= size_difference

        similarity = sys.maxsize
        if score < self.token_significant_size:
            # Not similar enough to be considered, ignore.
            similarity = 0
        else:
            # Above threashold, make a fraction between 0-1, scale to int.MaxValue.
            fscore = score / len(self.token_raw)

            if fscore >= .99:
                similarity = sys.maxsize
            else:
                similarity = int(fscore * sys.maxsize)

        return similarity


    def IsEqualTo(self, ref_token: TokenBase) -> bool:
        """
        Check if this token is equal to the reference token.
        """
        # Tokens are not the same if they are of different types.
        if not isinstance(ref_token, TokenReference):
            return False
        
        equal = len(self.token_raw) == len(ref_token.token_raw)
        if equal:
            for i in range(len(self.token_raw)):
                if not self.token_raw[i].IsEqualTo(ref_token.token_raw[i]):
                    equal = False
                    break
    
        return equal
    
        
