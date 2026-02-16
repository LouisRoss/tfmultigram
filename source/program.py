#from websockets.sync.client import connect

from multigram import MultiGram
from tokenstring import TokenString
from tokenstringembed import TokenStringEmbed
from tokensourcecsvstream import TokenSourceCSVStream
from tokensourcedataset import TokenSourceDataset
from settings import Settings, MultigramState
from tokentests import GenerateLikelyString, GenerateBestFitString, GenerateRandomSentence

def DisplayRelationships(multigram: MultiGram, token: TokenString):
    """
    Display the relationships of tokens in the multigram.
    This function iterates through the tokens and prints their relationships.
    """
    print(f"Token: '{token.token_raw}'")
    for distance in range(1, len(token.Connections) + 1):
        print(f"  Token: '{token.token_raw}' at distance {distance}:")
        for connection in token.Connections[distance - 1]:
            if connection.Strength > 1:
                print(f"    Connected to: {connection.FollowingToken.token_raw} with strength {connection.Strength}, softmax {connection.SoftmaxStrength} at distance {distance}")
    print()

    #for token in multigram.tokens:


def main():
    print("Starting Multigram processing...")
    # Use the Multigram to process tokens
    #with TokenSourceCSVStream('/log/syslog', 500) as token_source:
    with TokenSourceDataset("roneneldan/TinyStories", 200) as token_source:
        # Initialize the Multigram with a CSV token source
        multigram = MultiGram(token_source)

        #while multigram.next_token_index < 1000:
        while not multigram.input_source_complete:
            multigram.ReadTokenBehavior()

    print(f'Processed {multigram.CountUsedTokens()} tokens from the source.')
    multigram.Softmax()  # Apply softmax to all token connections after processing

    exampleToken = TokenString("Once")
    token = multigram.FindTokenLike(exampleToken)
    if token == None:
        print(f"Starting example token '{exampleToken.token_raw}' not found in the multigram.")
    else:
        DisplayRelationships(multigram, token)
        likely_string = GenerateLikelyString(multigram, token)
        print(f"Likely string generated: {likely_string}")

        # Print the results or perform further actions

        response = GenerateBestFitString(multigram, ["Once", "upon", "a"])
        print(f'Best fit response: {response}')

    starting_token = TokenString(Settings.StartOfSequenceTokenValue)
    root_token = starting_token.FindTokenIfSeen(tokens = multigram.tokens, threshold_score = 1.0)
    DisplayRelationships(multigram, root_token)

    random_sentence = GenerateRandomSentence(multigram)
    print(f'Randomly generated sentence 1: {random_sentence}')
    random_sentence = GenerateRandomSentence(multigram)
    print(f'Randomly generated sentence 2: {random_sentence}')
    random_sentence = GenerateRandomSentence(multigram)
    print(f'Randomly generated sentence 3: {random_sentence}')

    print("Multigram processing complete.")

if __name__ == "__main__":
    main()

