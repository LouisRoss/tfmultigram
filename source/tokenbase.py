import sys
from abc import ABC, abstractmethod
from settings import Settings, MultigramState

class TokenBase(ABC):
    """
    Base class for token management.
    This class provides a structure for managing tokens, including their creation,
    validation, and expiration.
    """

    def __init__(self, token_type: str):
        self.token_type = token_type

        self.OrgnizeSeen = False
        self.CurrentActivityFromPreviousTokens = 0.0
        self.CurrentActivityFromPreviousTokens = [0.0 for i in range(Settings.max_token_strength)]    
        self.CurrentStrength = 0


    def CaptureNewActivity(self) -> None:
        """
        """
        self.CurrentActivityFromPreviousTokens = [0.0 for i in range(Settings.max_token_strength)]    

    def create_token(self, user_id: str) -> str:
        """
        Create a new token for the given user ID.
        """
        raise NotImplementedError("This method should be overridden by subclasses.")

    def validate_token(self, token: str) -> bool:
        """
        Validate the given token.
        """
        raise NotImplementedError("This method should be overridden by subclasses.")

    def expire_token(self, token: str) -> None:
        """
        Expire the given token.
        """
        raise NotImplementedError("This method should be overridden by subclasses.")
    
    def CheckIfTokenSeen(self, ref_token: 'TokenBase', threshold_score: int = sys.maxsize) -> bool:
        """
        The reference token is considered 'seen' if it is similar
        enough to this token.  'Similar' enough means its similarity
        is greater than the threadshold passed in.
        ref_token: Reference token to compare with this token
        threashold_score: How similar is similar enough
        returns: True if similar enough to be called 'seen'
        """
        seen = False
        if self.CheckIfSimilar(ref_token, threshold_score) >= threshold_score:
            seen = True

        return seen
    
    def TriggerToken(self):
        """
        When a token is seen, its strength is set to the maximum possible.
        """
        self.CurrentStrength = Settings.max_token_strength


    def Tick(self) -> None:
        """
        After a token has been triggered, its strength decays linearly
        with every tick.
        """
        if self.CurrentStrength > 0:
            self.CurrentStrength -= 1


    @abstractmethod
    def CheckIfTokenSimilar(self, ref_token: 'TokenBase') -> int:
        """
        Return a similarity value between this token and the reference token.
        ref_token: Reference token to compare with this token
        returns: Similarity score between this token and the reference token
        """
        pass

    @abstractmethod
    def IsEqualTo(self, ref_token: 'TokenBase') -> bool:
        """
        /// Determine if the content of this token and the reference token are the same.
        ref_token: Reference token to compare with this token
        returns: True if the content of this token and the reference token are the same
        """
        pass

    @abstractmethod
    def GetAsString(self) -> str:
        """
        Return the string representation of this token.
        returns: String representation of this token
        """
        pass
