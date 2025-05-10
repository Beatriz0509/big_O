
from MockInputs.MockDefaultInputSizes import MockDefaultInputSizes

class MockDictInputs(MockDefaultInputSizes):

    def __init__(self, mockInputs):
        self.content = {}

    @staticmethod
    def get_random_data(mockInputs):
        return {str(i): i for i in range(mockInputs.min_n, mockInputs.max_n)}