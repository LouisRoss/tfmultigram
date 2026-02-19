import sys
from abc import ABC, abstractmethod
from settings import Settings, MultigramState

from tokensynapse import TokenSynapse

class TokenBase(ABC):
    """
    Base class for token management.
    This class provides a structure for managing tokens, including their creation,
    validation, and expiration.
    """

    def __init__(self, token_type: str):
        self.token_type = token_type

        self.start_of_sequence = False
        self.IntrinsicToken = False
        self.IntrinsicOperation = None
        self.OrgnizeSeen = False
        self.CurrentStrength = 0

        self.Connections = [[] for i in range(Settings.max_token_strength)]
        self.SoftmaxConnections = [[] for i in range(Settings.max_token_strength)]
        self.NomalizedConnections = [[] for i in range(Settings.max_token_strength)]
        self.ConnectionCount = [0 for i in range(Settings.max_token_strength)]
        self.TotalConnectionStrength = [0 for i in range(Settings.max_token_strength)]
        self.CurrentActivityFromPreviousTokens = [0.0 for i in range(Settings.max_token_strength)]


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
        if self.CheckIfTokenSimilar(ref_token) >= threshold_score:
            seen = True

        return seen
    
    def FindTokenIfSeen(self, tokens: list['TokenBase'], threshold_score: float = 1.0) -> 'TokenBase':
        """
        Examine all tokens for any that recognize this token.
        returns: The reference token if seen, None otherwise
        """
        inserted_token = None
        if tokens is not None:
            for i in range(len(tokens)):
                a_token = tokens[i]
                if a_token is not None:
                    # There is a used token here, allow it to recognize the reference.
                    if a_token.CheckIfTokenSeen(self, threshold_score):
                        # A token has recognized the reference, we don't need to add one.
                        inserted_token = a_token
                        break

        return inserted_token
    
    
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


    def IsRelatedTo(self, ref_token: 'TokenBase') -> bool:
        """
        The reference token passed in is related to this token
        if this token has been triggered recently, and its strength
        has not yet decayed to zero.
        /// </summary>
        ref_token: A token to check the relation to this token
        returns: True if the reference token is related to this token
        """
        related = False

        if self.CurrentStrength > 0 and ref_token.CurrentStrength > 0:
            related = True

        return related
    

    def BumpRelationship(self, ref_token: 'TokenBase') -> None:
        """
        Bump the relationship between this token and the reference token.
        This is done by increasing the strength of the connection.
        ref_token: A token to bump the relationship with
        """
        if ref_token.CurrentStrength != Settings.max_token_strength or \
            ref_token.CurrentStrength <= self.CurrentStrength or \
            self.CurrentStrength == 0:
                return
        

        distance = ref_token.CurrentStrength - self.CurrentStrength


    def BumpRelationship(self, ref_token: 'TokenBase', distance: int) -> None:
        """
        Bump the relationship between this token and the reference token.
        This is done by increasing the strength of the connection.
        ref_token: A token to bump the relationship with
        strength: The strength of the relationship
        distance: The distance between this token and the reference token
        """
        connected_synapse = None

        for synapse in self.Connections[distance - 1]:
            if synapse.FollowingToken.IsEqualTo(ref_token):
                connected_synapse = synapse
                break

        if connected_synapse is None:
            connected_synapse = TokenSynapse(ref_token, 0)
            self.Connections[distance - 1].append(connected_synapse)

        if not connected_synapse is None:
            connected_synapse.Strength += 1
            #self.Softmax()      # Do this for strict (and expensive) softmax updating.


    def Softmax(self) -> None:
        """
        Apply the softmax function to the given list of values.
        values: A list of values to apply the softmax function to
        returns: A list of values after applying the softmax function
        """
        for connectionsAtDistance in self.Connections:
            exp_values = [pow(2.71828, synapse.Strength) for synapse in connectionsAtDistance]
            sum_exp_values = sum(exp_values)
            for i in range(len(connectionsAtDistance)):
                if sum_exp_values == 0:
                    connectionsAtDistance[i].SoftmaxStrength = 0
                else:
                    connectionsAtDistance[i].SoftmaxStrength = exp_values[i] / sum_exp_values
    

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
