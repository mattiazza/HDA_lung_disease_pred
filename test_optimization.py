#!/usr/bin/env python3
"""
Quick test script for hyperparameter optimization.
Runs just a few trials to verify everything works.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from HDA_lung_disease_pred.scripts.optimization import optimize_hyperparameters


def quick_test():
    """Run a quick optimization test with minimal trials."""
    print("🧪 Running quick optimization test...")
    print("This will run only 3 trials with 10 epochs each to test the setup.")
    
    try:
        study = optimize_hyperparameters(
            n_trials=3,           # Very few trials for testing
            dataset_name="chestmnist_64",
            epochs=10,            # Few epochs for testing
            patience=5            # Early stopping patience
        )
        
        print("✅ Quick test completed successfully!")
        print(f"Best AUC from test: {study.best_value:.4f}")
        print("Best parameters:")
        for param, value in study.best_params.items():
            print(f"  {param}: {value}")
            
    except Exception as e:
        print(f"❌ Test failed: {str(e)}")
        raise


if __name__ == "__main__":
    quick_test()
