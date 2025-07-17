"""
"""
from tokenbase import TokenBase
from tokenstring import TokenString
from settings import Settings, MultigramState


class MultiGram:
    def __init__(self, source, threshold=0.5):
        self.state = MultigramState.IDLE
        self.current_line_estimated_count = 0
        self.current_token_estimated_count = 0

        self.token_source = source
        self.threshold_score = threshold
        self.recent = []
        self.eol_token_next = False

        self.tokens = [None for i in range(Settings.max_tokens)]    

        # Support for following behavior.
        self.recently_followed_tokens = []
        self.most_recently_followed_token = None
        self.eol_token_next = False

        self.Reset()


    def InputLineCount(self):
        if self.token_source is not None:
            return self.token_source.LineCount()
        
        return 0
    

    def Reset(self):
        """
        Reset the n-grams to an empty state.
        Allow external client callers to return the input state to original.
        """
        self.settle_count = 0
        self.eol_token_next = False
        self.input_source_complete = False
        self.recent = [None for i in range(Settings.max_token_strength)]


    def ReadTokenBehavior(self):
        """
        In order to imbue the Multigram with time-sensitivity, a tick must
        pass between each token's addition.  This allows connections to be 
        established between the token currently being added and tokens recently
        added.  Connections between tokens added at widely separated times cannot
        be established.
        """
        # If we are currently settling, just allow an idle tick.
        if self.settle_count > 0:
            self.Tick()

            self.settle_count -= 1
            if self.settle_count <= 0:
                self.ClearRecentMemory()
                self.settle_count = 0

        if self.token_source is None or self.settle_count != 0:
            return

        # If we are not settling (or have just finished settling), make connections.
        token_bytes = self.token_source.get_next()
        if token_bytes is not None:
            # Insert or find the token, strengthen existing connections or make new ones.
            self.ConnectToken(token_bytes, self.threshold_score)

            # Let the Multigram tick.
            self.Tick()

            # Detect end of line, establish a settle period to separate lines.
            if token_bytes is TokenString:
                if token_bytes.IsEndOfLine():
                    # Allow all token strengths to settle to zero.
                    self.settle_count = Settings.max_token_strength
        else:
            # We have no more input.
            if not self.input_source_complete:
                print("Input source complete, settling.")

            self.input_source_complete = True

    def FollowTokenBehavior(self, following_cutoff, next_layer):
        """
        Follow the token behavior for the next layer.
        This method is used to process tokens in a following context.
        """
        if self.token_source is None or self.input_source_complete:
            return

        if self.eol_token_next:
            self.eol_token_next = False

            eol_string = TokenString()
            eol_string.SetEndOfLine()
            if next_layer is not None:
                next_layer.Insert(eol_string)

            # Allow the map to settle.
            for _ in range(Settings.max_token_strength):
                self.Tick()

            # Return all accumulators to zero.
            self.SettleTokenActivity()


        # Evaluate current time-sequence input stream, determine expectations for following token.
        if self.most_recently_followed_token is not None:
            # If we have a most recently followed token, we can advance the recent memory.
            self.most_recently_followed_token.TriggerToken()

        # NOTE: I think this is for animation, deferred.
        #self.EvaluateTokenActivty()
        self.Tick()


        # Read a token from the token source.
        token_bytes = self.token_source.get_next()
        multigram_token = self.FindTokenLike(token_bytes)
        if multigram_token is not None:
            # If we found a token that matches, we can use it.
            self.DoFollowForToken(multigram_token, following_cutoff, next_layer)
        else:
            if not self.input_source_complete:
                print(f'Input complete with {self.CountUsedTokens()} tokens in {self.InputLineCount()} lines.')
                next_layer.MarkAsDone()

            self.input_source_complete = True



    def DoFollowForToken(self, token, following_cutoff, next_layer):
        # Most interesting cases happen only if we already have followed a token.
        if self.most_recently_followed_token is None:
            # First token in a line just starts a new collection.
            self.recently_followed_tokens = []

            if token is not None:
                self.recently_followed_tokens.append(token)

                # This token is now the recently seen one for next iteration.
                self.most_recently_followed_token = token
        else:
            # Follow the connection between the most recently followed token and the following one.
            connection_to_following_token = self.FollowToken(token)

            following_token = None
            if connection_to_following_token is not None:
                # We have a connection, follow it to the following token.
                following_token = connection_to_following_token.following_token

                # If we reached the end of a segment, build a new feature in the next layer, start a new segment.
                expected_next_tokens = []
                if self.ProcessFollowingTokens(connection_to_following_token, following_cutoff, expected_next_tokens):
                    if self.recently_followed_tokens is not None:
                        # Add _recentlyFollowedTokens as a single token to the next layer.
                        if next_layer is not None:
                            next_layer.insert(TokenReference(self.recently_followed_tokens) )

                        # Assume there will be no recently followed token for the next iteration, unless there is a FollowingToken.
                        self.recently_followed_tokens = None

                    # If there is a following token (no end of line), start a new feature collection with it.
                    if following_token is not None:
                        self.recently_followed_tokens = []
                        self.recently_followed_tokens.append(following_token)

            # Remember the most recently followed token for next follow.  Null is ok, for start of new line.
            self.most_recently_followed_token = following_token



    def ConnectToken(self, token, threshold_score):
        """
        As each token is read from the token source, it is connected into the
        Multigram here.  If the token is a duplicate of an existing token, the
        existing token is used.  Establish or reinforce connections with the
        inserted token and all recently added tokens.
        token: A token extracted from the input token source.
        thresholdScore: How similar must tokens be to be considered the same.
        returns: The token in the Multigram after connecting.  This may be the token passed in or an identical one already in the Multigram.
        """

        # Add this token to the Multigram.  This will find an identical token already
        # in the Multigram if it exists, or return the token passed in if an identical one does not exist.
        # Also, the token returned will have been triggered.
        inserted_token = self.AddToken(token, threshold_score)

        if inserted_token is not None:
            # Examine all recently-seen tokens, bump the relationship of the new-or-found token with them all.
            for i in range(len(self.recent)):
                if self.recent[i] is not None:
                    # Bump the relationship of the recent token with the inserted token, at the given strength.
                    self.recent[i].BumpRelationship(inserted_token, i + 1)

            # Recent memory is a shift register, with the oldes falling off the end, while the new one is inserted.
            self.AdvanceRecentMemory(inserted_token)

        return inserted_token


    def AddToken(self, token, threshold_score):
        """
        Find a token that recognizes this reference token, or make a new one.  
        Excite the new-or-found token.
        token: A token to add to the Multigram.
        thresholdScore: How similar tokens must be to be considered the same.
        returns: The added token or one found in the Multigram already, identical with the added token.
        """
        seen = False
        first_unused_token = -1

        # Examine all tokens in the map.  For any that recognize the reference token,
        # let them set their excitation level.  While in the loop, capture the index
        # of the first unused token, in case none recognizes the reference, and we need to add one.
        inserted_token = None
        for i in range(len(self.tokens)):
            a_token = self.tokens[i]
            if a_token is not None:
                # There is a used token here, allow it to recognize the reference.
                if a_token.CheckIfTokenSeen(a_token, threshold_score):
                    # We found an existing token, trigger it.
                    a_token.TriggerToken()

                    # A token has recognized the reference, we don't need to add one.
                    seen = True
                    inserted_token = a_token
                    break

        # If no token recognized the reference, add a new token here.
        if not seen and first_unused_token == -1:
            # Keep reference to the new token here.
            self.tokens[first_unused_token] = token
            inserted_token = token

            # Trigger this new token.
            inserted_token.TriggerToken()

        return inserted_token


    def FindTokenLike(self, token):
        """
        Given a token, examine all tokens in this multigram, and find the
        one that is equal to the target.
        token: A target token to search for.
        returns: The token in the multigram like the target token, or null if none exits.
        """
        found_token = None

        for a_token in self.tokens:
            if a_token is not None:
                if a_token.IsEqualTo(token):
                    found_token = a_token
                    break

        return found_token
    

    def FollowToken(self, next_token):
        if self.most_recently_followed_token is None or next_token is None:
            return None
        
        connection_to_following_tokan = None

        for connection in self.most_recently_followed_token.connections:
            if connection.distance == 1 and connection.following_token.IsEqualTo(next_token):
                connection_to_following_tokan = connection
                break

        return connection_to_following_tokan
    
        
    def SettleTokenActivity(self):
        """
        Settle the activity of tokens.
        This method is used to allow tokens to settle after a period of activity.
        """
        for token in self.tokens:
            if token is not None:
                token.CaptureNewActivity()



    def Tick(self):
        """
        Perform a tick operation on the Multigram.
        Advance the clock one tick for all tokens in the map simultaneously.
        """
        # Process recent tokens and update their strengths.
        for token in self.tokens:
            if token is not None:
                token.Tick()

    def ClearRecentMemory(self):
        """
        Clear the recent memory of tokens.
        This is used to reset the Multigram's state after a settle period.
        """
        self.recent = [None for i in range(Settings.max_token_strength)]

    
    def AdvanceRecentMemory(self, token):
        """
        The recent memory array is used as a shift register.  Advancing
        the recent memory means shifting out the oldest token, and adding
        the specified token in at the front.
        token: The new token to add at the front of the shift register.
        """
        self.recent.pop(0)
        self.recent.append(token)

    def CountUsedTokens(self):
        """
        All tokens are stored in an array of fixed size, with nulls
        indicating unused token storage.  Count used tokens in the Multigram.
        returns: The count of non-null tokens in the array.
        """
        count = 0
        for token in self.tokens:
            if token is not None:
                count += 1

        return count
    
