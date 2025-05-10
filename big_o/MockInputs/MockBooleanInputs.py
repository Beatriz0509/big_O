import random
from MockInputs.MockDefaultInputSizes import MockDefaultInputSizes

class MockBooleanInputs(MockDefaultInputSizes):

    def __init__(self, mockInputs):
        self.content = self.get_random_data()

    @staticmethod
    def get_random_data():
        # Returns a random boolean
        return random.choice([True, False])