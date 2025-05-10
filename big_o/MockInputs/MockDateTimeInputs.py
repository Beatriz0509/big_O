from datetime import datetime
import random
from MockInputs.MockDefaultInputSizes import MockDefaultInputSizes

class MockDateTimeInputs(MockDefaultInputSizes):

    datetime_inputs = [
        datetime(2025, 1, 1, 12, 0, 0),  # A specific datetime
        datetime.now(),  # Current time
        datetime(2020, 5, 20, 18, 30, 0)  # Another sample datetime
    ]

    def __init__(self, mockInputs):
        self.content = self.get_random_data()

    @staticmethod
    def get_random_data():
        # Returns a random datetime from the list
        return random.choice(MockDateTimeInputs.datetime_inputs)