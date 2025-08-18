import re
from tokenbase import TokenBase
from tokenstringembed import TokenStringEmbed
from tokentimestamp import TokenTimestamp
from tokensourcebase import TokenSourceBase

timestamp_pattern = re.compile(r'^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}.\d{6}[\+|\-]\d{2}:\d{2}')

class TokenSourceCSVStream(TokenSourceBase):
    """
    Token source for reading tokens from a CSV stream.
    This class implements the abstract methods defined in TokenSourceBase.
    """

    def __init__(self, filename, max_lines=0):
        super().__init__()
        self.log_filename = filename
        self.max_lines = max_lines

        self.Reset()


    def __enter__(self):
        self.line_count = 0
        self.current_line = None

        self.Reset()

        return self
    
    def __exit__(self, exc_type, exc_value, traceback):
        if self.istream:
            self.istream.close()
            self.istream = None


    def IsInputAvailable(self) -> bool:
        """
        Overridden method to check if the input stream is available.
        Returns True if the input stream is open and not closed.
        """
        return self.istream is not None and not self.istream.closed

    def GetLineCount(self) -> int:
        """
        Overridden method counts the number of lines read so far from the file.
        """
        return self.line_count

    def Reset(self) -> None:
        """"
        Reset the input stream to its beginning, set
        internal state as if nothing has been read.
        """
        self.istream = open(self.log_filename, 'r')

        self.last_line_read = None
        self.end_of_stream = False
        self.line_count_read = 0
        self.current_line = None



    def GetNext(self) -> TokenBase:
        """
        Overridden methos to read a new token from the input stream and return that token.
        Return null if we have reached the end of the stream and no
        new token is available.

        returns: The next token from the stream, or null if no new token is available.
        """
        return self.NextToken()
        
    def NextToken(self) -> TokenBase:
        """
        Returns the next token from the input stream.
        If the end of the stream is reached, returns None.
        """
        if not self.IsInputAvailable() or self.end_of_stream:
            return None
        
        
        next_token = self.PopTokenFromInput()

        if next_token is None and (self.last_line_read is None or len(self.last_line_read) == 0):
            self.ReadNextLine()
            if self.last_line_read is None:
                self.end_of_stream = True
                return None
            
            next_token = self.PopTokenFromInput()

        return next_token


    def ReadNextLine(self) -> None: 
        line = self.istream.readline()
        self.line_count_read += 1
        if self.line_count_read % 100 == 0:
            print()
            print('************************************************')
            print(f"Read {self.line_count_read} lines from {self.log_filename}")
            print('************************************************')

        if not line or (self.max_lines > 0 and self.line_count_read >= self.max_lines):
            print()
            print('************************************************')
            print(f"Read {self.line_count_read} lines from {self.log_filename}")
            print('************************************************')
            self.last_line_read = None
            return

        self.last_line_read = re.split("\s", line.strip())


    def PopTokenFromInput(self) -> TokenBase:
        """
        The caller guarantees that the list of strings
        LastLineRead exists, and contains whatever tokens remain
        from the last line read.  This includes the cases where
        the last line read was empty, and where previous calls
        to this method have exhausted all tokens from LastLineRead.
        When this method is called with the resulting empty LastLineRead,
        it generates a TokenStringEmbed with .EndOfLine set, and nulls LastLineRead.

        returns: The next token from LastLineRead, or null if none exist.
        """
        next_token = None

        if self.last_line_read is not None:
            if len(self.last_line_read) != 0:
                token_value = self.last_line_read.pop(0)
                if re.match(timestamp_pattern, token_value):
                    # If the token is a timestamp, create a TokenTimestamp object
                    next_token = TokenTimestamp(token_value)
                else:
                    # Otherwise, create a TokenBase object
                    next_token = TokenStringEmbed(token_value)
            else:
                next_token = TokenStringEmbed('')
                next_token.end_of_line = True

                self.last_line_read = None
            
        return next_token
    