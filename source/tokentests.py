from multigram import MultiGram
from tokenbase import TokenBase
from tokenstringembed import TokenStringEmbed
from tokenstring import TokenString
from settings import Settings

def FindMostLikelyNextToken(multigram: MultiGram, token_history: list[TokenString], threshold: int = 1) -> TokenString:
    """
    Find the most likely next token based on the current token's relationships.
    This function uses the token's connections to determine the next likely token.
    """
    likely_tokens = {}

    history_length = len(token_history) if len(token_history) < Settings.max_token_strength else Settings.max_token_strength
    for distance in range(1, history_length + 1):
        token = token_history[-distance]
        distance_multiplier = len(token_history) - distance + 1

        for connection in token.Connections:
            #print(f'Examining connection from {token.token_raw} to {connection.FollowingToken.token_raw if connection.FollowingToken is not None else "<None>"} with strength {connection.Strength} at distance {connection.Distance}')
            if connection.Distance == distance:
                if distance > 1 and connection.FollowingToken in likely_tokens:
                    likely_tokens[connection.FollowingToken] += connection.Strength * distance_multiplier
                elif distance == 1 and connection.FollowingToken not in likely_tokens:
                    likely_tokens[connection.FollowingToken] = connection.Strength * distance_multiplier

    print()
    print('*************************************************')
    print(f'Finding next likely token out of {len(likely_tokens)} possibilities after ' + ' -> '.join(token_history[i].token_raw for i in range(0, len(token_history))))
    #for token, strength in likely_tokens.items():
    #    print(f'  Possible next token: {token.token_raw} with strength {strength}')
    print('*************************************************')
    print()

    likely_token = None
    max_strength = 0
    for token, strength in likely_tokens.items():
        if strength > max_strength and strength >= threshold:
            likely_token = token
            max_strength = strength

    print(f'Most likely next token for "{token_history[-1].token_raw}" is {likely_token.token_raw if likely_token is not None else "<None>"} with strength {max_strength}')

    return likely_token


def GenerateLikelyString(multigram: MultiGram, token: TokenString) -> str:
    """
    Generate a likely string based on the token and its relationships.
    This function uses the token's connections to build a string representation.
    """
    result = []

    while token is not None:
        result.append(token)
        #print(token.token_raw)
        token = FindMostLikelyNextToken(multigram, result)

    string_result = [t.token_raw if t is not None else '<None>' for t in result]
    return ' -> '.join(string_result)


def dot(va, vb):
    return sum(a * b for a, b in zip(va, vb))

def FindBestFitNextToken(multigram: MultiGram, token: TokenString) -> TokenString:
    """
    Find the best fit next token based on the current token's relationships.
    This function uses the token's connections to determine the next best fit token.
    """
    return token.FindTokenIfSeen(tokens = None, threshold_score = 0.0)


def ResolveQueryTokens(multigram: MultiGram, query_tokens: list['str']) -> list['str']:
    """
    Resolve a list of query tokens to their corresponding TokenString objects.
    This function attempts to find the best match for each token in the provided list.
    """
    result = []

    for token_str in query_tokens:
        token_embed = TokenString(token_str)
        resolved_token = token_embed.FindTokenIfSeen(tokens = None, threshold_score = 0.0)
        result.append(resolved_token.token_raw if resolved_token is not None else '<None>')

    return result


def GenerateBestFitString(multigram: MultiGram, tokens: list['str']) -> list['str']:
    """
    Generate the best fit string based on the token and its relationships.
    This function uses the token's connections to build a string representation.
    """
    result = []

    root_token = TokenString(tokens[0])
    print(f'GenerateBestFitString Finding best fit for root token "{root_token.token_raw}"')
    token = root_token.FindTokenIfSeen(tokens = multigram.tokens, threshold_score = 1.0)
    print(f'Best fit for root token "{root_token.token_raw}" is "{token.token_raw if token is not None else "<None>"}"')
    result.append(token)

    for next_prompt in tokens[1:]:
        token_prompt = TokenString(next_prompt)
        prompt_possible = [c.FollowingToken for c in token.Connections if c.Distance == 1 and c.FollowingToken is not None]
        print(f'Finding possible next prompts for {token.token_raw}: {[p.token_raw for p in prompt_possible]}')
        if isinstance(token_prompt, TokenStringEmbed):
            prompt_similarities = [dot(token_prompt.embedding, p.embedding) for p in prompt_possible]
            max_similarity = max(prompt_similarities) if len(prompt_similarities) > 0 else 0.0
            index = prompt_similarities.index(max_similarity) if max_similarity > 0.0 else -1

            if index > -1 and len(prompt_similarities) > 0:
                token = prompt_possible[index]
                result.append(token)
        else:
            for possible in prompt_possible:
                if len(possible.token_raw) == len(token_prompt.token_raw):
                    token = possible
                    result.append(token)
                    break


    while token is not None:
        print(f'Extending best fit with {token.token_raw}')
        token = FindMostLikelyNextToken(multigram, result, threshold=0)
        result.append(token)

    string_result = [t.token_raw if t is not None else '<None>' for t in result]
    return ' -> '.join(string_result)
