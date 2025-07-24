from abc import ABC, abstractmethod
from tokenbase import TokenBase

class TokenSourceBase(ABC):
    """
    Base class for token sources.
    This abstract class provides the formal definition of a source of tokens.
    """
    
    def __init__(self):
        pass

    @abstractmethod
    def IsInputAvailable(self) -> bool:
        """
        Read-only end-of-source indication, must be overridden.
        """
        pass

    @abstractmethod
    def GetLineCount(self) -> int:
        """
        Returns the number of lines available in the token source.
        """
        pass

    @abstractmethod
    def Reset(self) -> None:
        """
        Restart token source from the beginning, if possible.
        May be overridden.
        """
        pass

    @abstractmethod
    def GetNext(self) -> TokenBase:
        """
        Returns the next token from the source.
        If no more tokens are available, returns None.
        Must be overridden.
        """
        pass
    
