import sys
import datetime
from tokenbase import TokenBase

class TokenTimestamp(TokenBase):
    """
    Class to handle token timestamps.
    """
    def __init__(self, timestamp: datetime.datetime = None):
        super().__init__('TokenTimestamp')
        if timestamp is None:
            self.token_raw = datetime.datetime.MINYEAR
        else:
            self.token_raw = timestamp

    def SetTime(self, timestamp: str) -> None:
        """
        Set the timestamp for this token from a string in ISO format.
        """
        self.token_raw = datetime.fromisoformat(timestamp)

    def CheckIfTokenSimilar(self, ref_token: 'TokenBase') -> int:
        """
        Return a similarity value between this token and the reference token.
        ref_token: Reference token to compare with this token
        returns: Similarity score between this token and the reference token
        """
        if not isinstance(ref_token, TokenTimestamp):
            return 0

        return sys.maxsize

    def IsEqualTo(self, ref_token: 'TokenBase') -> bool:
        """
        /// Determine if the content of this token and the reference token are the same.
        ref_token: Reference token to compare with this token
        returns: True if the content of this token and the reference token are the same
        """
        if not isinstance(ref_token, TokenTimestamp):
            return False

        return True

    def GetAsString(self) -> str:
        """
        Return the string representation of this token.
        returns: String representation of this token
        """
        return self.token_raw.isoformat() if self.token_raw else datetime.datetime.MINYEAR.isoformat()
    
