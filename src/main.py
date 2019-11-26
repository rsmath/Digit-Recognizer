"""
This module is just to run the model in a clean environment with new data
"""

from src.model import VanillaNN


model = VanillaNN(print_cost=True)

if __name__ == "__main__":
    model.train()






