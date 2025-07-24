
class TokenSynapse:
    """
    This class implements the connection between a token and the
    tokens that follow it in sequence.
    For simplicity, refer to 'this token' as the token that
    owns this synapse.
    NOTE: after must be a TokenBase object.
    """
    def __init__(self, after, strength:int=0, distance:int=0):
        """
         A synapse must be provided with the following
        token, the initial strength, and the distance between this token
        and the following token.

        after: A token that follows this token in the sequence
        strength: Initial strength of the relationship
        distance: How far the following token follows this token
        """
        self.FollowingToken = after
        self.Strength = strength
        self.Distance = distance


    def Dump(self) -> None:
        """
        Print a string representation of this synapse.
        """
        print(f"   TokenSynapse connects preceding token to {self.FollowingToken.GetAsString()} with a strength of {self.Strength} at a distance of {self.Distance})")

       
