import random
from MockInputs.MockDefaultInputSizes import MockDefaultInputSizes

class MockStringInputs(MockDefaultInputSizes):

    def __init__(self, mockInputs):
        self.content = self.get_random_data(mockInputs)

    @staticmethod
    def get_random_data(mockInputs):
        length = random.randint(mockInputs.min_n, mockInputs.max_n)
        # Returns a random string of lowercase letters with a random length
        return ''.join(random.choices('abcdefghijklmnopqrstuvwxyz', k=length))