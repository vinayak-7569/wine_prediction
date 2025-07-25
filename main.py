import yaml
import os
from src.production_model import create_production_model

def main():
    print("\n Starting Wine Quality Prediction Pipeline...\n")

    # Only run production model creation (self-contained function)
    create_production_model()

if __name__ == '__main__':
    main()
