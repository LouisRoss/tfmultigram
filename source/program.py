from multigram import MultiGram
from tokensourcecsvstream import TokenSourceCSVStream


def main():

    # Use the Multigram to process tokens
    with TokenSourceCSVStream('/log/syslog', 1000) as token_source:
        # Initialize the Multigram with a CSV token source
        multigram = MultiGram(token_source)

        #while multigram.next_token_index < 1000:
        while not multigram.input_source_complete:
            multigram.ReadTokenBehavior()

    print(f'Processed {multigram.CountUsedTokens()} tokens from the source.')

    # Print the results or perform further actions
    print("Multigram processing complete.")

if __name__ == "__main__":
    main()

