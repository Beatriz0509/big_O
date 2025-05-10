import random
from MockInputs.MockDefaultInputSizes import MockDefaultInputSizes

class MockListInputs(MockDefaultInputSizes):

    def __init__(self, mockInputs):
        self.content = []
    
    @staticmethod
    def restrictions(n_list):
        return [abs(n) for n in n_list]
    
    @staticmethod
    def get_random_data(mockInputs):
        return [random.randint(mockInputs.min_n, mockInputs.max_n) for _ in range(random.randint(mockInputs.min_n, mockInputs.max_n))]

