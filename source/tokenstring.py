import sys
from tokenbase import TokenBase

class TokenString(TokenBase):
    """
    TokenString is a concrete implementation of the TokenBase class.
    It represents a token that is a string and provides specific implementations
    for the abstract methods defined in TokenBase.
    """

    def __init__(self, value: str):
        super().__init__()
        self.token_raw = value
        self.end_of_line = False

    
    # Concrete implementation of abstract methods from TokenBase
    def CheckIfTokenSimilar(self, ref_token: TokenBase) -> int:
        """
        Return a measure of the similarity between this token and refToken.
        As a string, we choose a binary answer of 0 (not the same) or
        sys.maxsize (identical).
        ref_token: Reference token to compare with this token
        returns: Similarity measure between 0 and sys.maxsize
        """
        if not ref_token is TokenString:
            return 0
        
        return sys.maxsize if self.GetAsString() == ref_token.GetAsString() else 0
    
    def IsEqualTo(self, ref_token):
        """
        Boolean equality.  True if this token and refToken encode the same value.
        For strings, this gives the same results as CheckIfTokenSimilar,
        but other types may work differently.
        /// </summary>
        ref_token: A reference token to compare with this token
        returns: True if tokens encode the same value, false otherwise
        """
        if not ref_token is TokenString:
            return False

        return self.GetAsString() == ref_token.GetAsString()
    
    def GetAsString(self) -> str:
        """
        For logging and analysis, get this token as a string.
        For string type, this is easy.
        returns: The string value this token encodes.
        """
        if self.end_of_line:
            return "<eol>"
        
        return self.token_raw
    
