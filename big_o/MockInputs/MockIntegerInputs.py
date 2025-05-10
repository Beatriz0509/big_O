import random
from MockInputs.MockDefaultInputSizes import MockDefaultInputSizes

class MockIntegerInputs(MockDefaultInputSizes):

    @staticmethod
    def get_random_data(mockInputs):
        """
        Generate a random integer based on the given size n, 
        where n determines the range for the random integer.

        Args:
        - n (int): The size parameter to influence the range of random integers.

        Returns:
        - int: A random integer within a range determined by n.
        """
        
        # The generated number is scaled based on the size n
        # For example, the range for the random integer will be (1, n)
        return random.randint(mockInputs.min_n, mockInputs.max_n)
    
    @staticmethod
    def restrictions(n_list):
        return [int(n) for n in n_list]
