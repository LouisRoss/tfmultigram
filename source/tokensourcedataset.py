import re
from datasets import load_dataset

from settings import Settings, TokenSourceFlags
from tokenbase import TokenBase
from tokenstring import TokenString
#from tokenstringembed import TokenStringEmbed
from tokentimestamp import TokenTimestamp
from tokensourcebase import TokenSourceBase

sentences_pattern = re.compile(r"[\w+\s]+.")
words_pattern = re.compile(r"([\w]+)(.)")

class TokenSourceDataset(TokenSourceBase):
    """
    Token source for reading tokens from a hugging face dataset.
    This class implements the abstract methods defined in TokenSourceBase.
    """

    def __init__(self, datasetname, max_lines=0):
        super().__init__()
        self.datasetname = datasetname
        self.max_lines = max_lines

        self.Reset()


    def __enter__(self):
        self.line_count = 0
        self.current_line = None

        self.Reset()

        return self
    
    def __exit__(self, exc_type, exc_value, traceback):
        pass


    def IsInputAvailable(self) -> bool:
        """
        Overridden method to check if the input stream is available.
        Returns True if the input stream is open and not closed.
        """
        if self.current_delimiter != ' ':
            return True
        
        if len(self.current_sentence) > 0:
            return True

        if len(self.story) > 0:
            return True

        if self.current_story < self.max_story:
            return True

        return False
    
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
        self.dataset = load_dataset(self.datasetname)

        # A story is a collection of sentences, and a sentence is a collection of 
        # tuples of (word, punctuation) pairs.  We will read the dataset one sentence at a time, 
        # and then read the tokens from each sentence one at a time.
        self.story = []
        self.current_sentence = []
        self.current_delimiter = ' '

        self.max_story = len(self.dataset['train']) if self.max_lines <= 0 else self.max_lines
        self.current_story = 0
        self.line_count = 0



    def GetNext(self, flags: int = 0) -> TokenBase:
        """
        Overridden method to read a new token from the input stream and return that token.
        Return null if we have reached the end of the stream and no
        new token is available.

        returns: The next token from the stream, or null if no new token is available.
        """

        # If requested, return a token indicating the start of a sequence.
        if flags & TokenSourceFlags.Flag_StartOfSequence:
            token = TokenString(Settings.StartOfSequenceTokenValue)
            token.start_of_sequence = True
            return token

        token = self.GetTokenFromLine()
        if token is not None:
            return token

        token = self.GetLineFromStory()
        if token is not None:
            return token

        token = self.GetStoryFromDataset()
        return token

    def GetStoryFromDataset(self) -> TokenBase:
        """
        Returns the next story from the dataset being read.
        If the end of the dataset is reached, returns None.
        """
        if self.current_story < self.max_story:
            tiny_story = self.dataset['train'][self.current_story]['text']
            self.current_story += 1
            print()
            print('************************************************')
            print(f"Read {self.current_story} stories of {self.max_story} from {self.datasetname}")
            print('************************************************')

            # Split the current line into sentences, and then split each sentence into words and delimiters
            self.story = sentences_pattern.findall(tiny_story)
            return self.GetLineFromStory()
        else:
            return None
        
    def GetLineFromStory(self) -> TokenBase:
        """
        Returns the next line from the current story being read.
        If the end of the story is reached, returns None.
        """
        if len(self.story) > 0:
            sentence = self.story.pop(0)
            self.current_sentence = words_pattern.findall(sentence)
            self.line_count += 1
            if self.line_count % 100 == 0:
                print()
                print('************************************************')
                print(f"Read {self.line_count} lines from {self.datasetname}")
                print('************************************************')

            return self.GetTokenFromLine()
        else:
            return None
        
    def GetTokenFromLine(self) -> TokenBase:
        """
        Returns the next token from the current line being read.
        If the end of the line is reached, returns None.
        """
        if self.current_delimiter != ' ':
            # If the current delimiter is not a space, return it as a token
            token = TokenString(self.current_delimiter)
            if self.current_delimiter == '.':
                token.end_of_line = True
            self.current_delimiter = ' '
            return token
        
        if len(self.current_sentence) > 0:
            word, self.current_delimiter = self.current_sentence.pop(0)
            token = TokenString(word)
            return token
        else:
            return None
                    

