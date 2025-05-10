import random
from MockInputs.MockDefaultInputSizes import MockDefaultInputSizes

class MockFloatInputs(MockDefaultInputSizes):

    def __init__(self, mockInputs):
        self.content = self.get_random_data(mockInputs)

    @staticmethod
    def get_random_data(mockInputs):
        return random.uniform(mockInputs.min_n, mockInputs.max_n)