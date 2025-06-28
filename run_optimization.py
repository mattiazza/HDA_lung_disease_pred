#!/usr/bin/env python3
"""
Example script for running hyperparameter optimization.

This script demonstrates how to use the optimization.py module
to find the best hyperparameters for your CNN model.
"""

from HDA_lung_disease_pred.scripts.optimization import (
    optimize_hyperparameters, 
    train_best_model
)
import argparse


def main():
    parser = argparse.ArgumentParser(description='CNN Hyperparameter Optimization')
    parser.add_argument('--trials', type=int, default=20, 
                       help='Number of optimization trials (default: 20)')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Max epochs per trial (default: 50)')
    parser.add_argument('--dataset', type=str, default='chestmnist_64',
                       help='Dataset name (default: chestmnist_64)')
    parser.add_argument('--patience', type=int, default=10,
                       help='Early stopping patience (default: 10)')
    parser.add_argument('--final-epochs', type=int, default=100,
                       help='Epochs for final model training (default: 100)')
    parser.add_argument('--no-final-training', action='store_true',
                       help='Skip final model training with best params')
    
    args = parser.parse_args()
    
    print("🔧 CNN Hyperparameter Optimization")
    print("=" * 50)
    print(f"Configuration:")
    print(f"  - Trials: {args.trials}")
    print(f"  - Max epochs per trial: {args.epochs}")
    print(f"  - Dataset: {args.dataset}")
    print(f"  - Patience: {args.patience}")
    print(f"  - Final training epochs: {args.final_epochs}")
    print("=" * 50)
    
    # Run optimization
    study = optimize_hyperparameters(
        n_trials=args.trials,
        dataset_name=args.dataset,
        epochs=args.epochs,
        patience=args.patience
    )
    
    # Train final model with best hyperparameters
    if not args.no_final_training:
        print("\n" + "=" * 50)
        print("🏁 Training final model with best hyperparameters...")
        
        final_model = train_best_model(
            study=study,
            dataset_name=args.dataset,
            epochs=args.final_epochs,
            save_path="models/cnn_baseline_optimized_final.keras"
        )
        
        print("✅ Optimization and training completed!")
        print(f"📁 Final model saved to: models/cnn_baseline_optimized_final.keras")
    else:
        print("✅ Optimization completed! Skipping final training.")
    
    # Print summary
    print("\n" + "🎯 OPTIMIZATION SUMMARY" + "🎯")
    print("=" * 50)
    print(f"Best validation AUC: {study.best_value:.4f}")
    print("Best hyperparameters:")
    for param, value in study.best_params.items():
        print(f"  • {param}: {value}")


if __name__ == "__main__":
    main()
