#from websockets.sync.client import connect

from multigram import MultiGram
from tokenstring import TokenString
from tokenstringembed import TokenStringEmbed
from tokensourcecsvstream import TokenSourceCSVStream
from tokensourcedataset import TokenSourceDataset
from settings import Settings, MultigramState
from tokentests import GenerateLikelyString, GenerateBestFitString

def DisplayRelationships(multigram: MultiGram, token: TokenString):
    """
    Display the relationships of tokens in the multigram.
    This function iterates through the tokens and prints their relationships.
    """
    print(f"Token: '{token.token_raw}'")
    for distance in range(Settings.max_token_strength):
        print(f"  Token: '{token.token_raw}' at distance {distance + 1}:")
        for connection in token.Connections:
            if connection.Distance == distance + 1 and connection.Strength > 1:
                print(f"    Connected to: {connection.FollowingToken.token_raw} with strength {connection.Strength} at distance {connection.Distance}")
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

    print("Multigram processing complete.")

if __name__ == "__main__":
    main()

