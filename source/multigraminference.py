import numpy as np
from settings import Settings
from tokenbase import TokenBase

# NOTE: This is out of date, as token.Connections is now a list of lists, where each inner list corresponds to a specific distance. 
# The code will need to be updated to reflect this change.
class MultigramInfer:
    def __init__(self):
        self.Predictions = []

    def AddPrediction(self, token: TokenBase) -> None:
        token_prediction = [[] for i in range(Settings.max_token_strength)]

        for connection in token.Connections:
            token_prediction[connection.Distance - 1].append(connection)

        self.Predictions.append(token_prediction)

    def Predict(self) -> list:
        if not self.Predictions:
            return []

        for i in range(len(self.Predictions)):
            distance = len(self.Predictions) - i - 1
            connections_at_distance = np.array(self.Predictions[distance])

    def PrintPredictionsForDistance(self, token: TokenBase, distance: int) -> None:
        if distance < 1 or distance > Settings.max_token_strength:
            print(f"Distance {distance} is out of bounds.")
            return
        
        print(f"Predictions for token '{token.GetAsString()}' at distance {distance}:")

        connections = token.Connections
        filtered_connections = [conn for conn in connections if conn.Distance == distance]

        if not filtered_connections:
            print(f"No connections found at distance {distance}.")
            return

        print(f"Connections at distance {distance}:")
        for conn in filtered_connections:
            print(f" - {conn.FollowingToken.GetAsString()} (Strength: {conn.Strength})")


    def PrintPredictions(self) -> None:
        if not self.Predictions:
            print("No predictions available.")
            return

        distance = 1
        for distance in range(1, Settings.max_token_strength + 1):
            print(f"\nPredictions at distance {distance}:")
            for token_prediction in self.Predictions:
                connections_at_distance = token_prediction[distance - 1]
                if connections_at_distance:
                    print(f"Connections: {[conn.FollowingToken.GetAsString() for conn in connections_at_distance]}")
                else:
                    print("No connections at this distance.")

            distance += 1
