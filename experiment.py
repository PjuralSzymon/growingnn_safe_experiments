import os
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import kerne
from datetime import datetime
import sys

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Define datasets for experiments
DATASETS = [
    # Small dataset with few classes (2D shape)
    'ArrowHead',      # 3 classes, 211 samples, shape: (211, 251)
    
    # Medium dataset with many classes (2D shape)
    'Adiac',          # 37 classes, 781 samples, shape: (781, 176)
    
    # Large dataset with few classes (1D shape)
    'ElectricDevices',  # 7 classes, 16637 samples, shape: (16637, 96)
    
    # Dataset with many samples but few features (1D shape)
    'FordA',          # 2 classes, 4921 samples, shape: (4921, 500)
    
    # Dataset with balanced classes and medium size (2D shape)
    'SwedishLeaf',    # 15 classes, 1125 samples, shape: (1125, 128)
    
    # Dataset with very long sequences (1D shape)
    'UWaveGestureLibraryX',  # 8 classes, 9458 samples, shape: (9458, 315)
    
    # Dataset with very short sequences (1D shape)
    'Wine',           # 2 classes, 111 samples, shape: (111, 234)
]

# Hyperparameters
WORD_LENGTH = 4
EMBEDDING_DIM = 4
EPOCHS = 5  # Reduced from 10
GENERATIONS = 3  # Reduced from 5

def run_comparison():
    """Run comparison between SAFE and SAFE with GrowingNN."""
    # Create a unique experiment folder
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_folder = f"experiments/exp_{timestamp}"
    os.makedirs(experiment_folder, exist_ok=True)
    
    # Store results
    results = []
    
    # Run experiments
    for dataset in DATASETS:
        # First try GrowingNN
        model_type = "SAFE with GrowingNN"
        logging.info(f"Running experiment: Dataset={dataset}, Model={model_type}")
        
        try:
            result = kerne.train_time_series_classifier(
                dataset_name=dataset,
                word_length=WORD_LENGTH,
                embedding_dim=EMBEDDING_DIM,
                epochs=EPOCHS,
                use_growing_nn=True
            )
            
            # Extract accuracy from result
            if isinstance(result, dict):
                accuracy = result.get('accuracy')
                if accuracy is None:
                    raise ValueError("No accuracy metric found in GrowingNN results")
            else:
                accuracy = result
                
            results.append({
                'dataset': dataset,
                'model': model_type,
                'accuracy': accuracy,
                'error': None
            })
            
        except Exception as e:
            error_msg = str(e)
            logging.error(f"GrowingNN training failed for {dataset}: {error_msg}")
            import traceback
            logging.error("Full stack trace:")
            logging.error(traceback.format_exc())
            # Exit the script on GrowingNN failure
            sys.exit(1)
        
        # Then try standard SAFE
        model_type = "SAFE"
        logging.info(f"Running experiment: Dataset={dataset}, Model={model_type}")
        
        try:
            result = kerne.train_time_series_classifier(
                dataset_name=dataset,
                word_length=WORD_LENGTH,
                embedding_dim=EMBEDDING_DIM,
                epochs=EPOCHS,
                use_growing_nn=False
            )
            
            # Extract accuracy from result
            if isinstance(result, dict):
                accuracy = result.get('accuracy')
                if accuracy is None:
                    raise ValueError("No accuracy metric found in SAFE results")
            else:
                accuracy = result
                
            results.append({
                'dataset': dataset,
                'model': model_type,
                'accuracy': accuracy,
                'error': None
            })
            
        except Exception as e:
            error_msg = str(e)
            logging.error(f"SAFE training failed for {dataset}: {error_msg}")
            import traceback
            logging.error("Full stack trace:")
            logging.error(traceback.format_exc())
            # Exit the script on SAFE failure
            sys.exit(1)
    
    # Convert results to DataFrame
    results_df = pd.DataFrame(results)
    
    # Save results
    csv_path = os.path.join(experiment_folder, "results.csv")
    results_df.to_csv(csv_path, index=False)
    logging.info(f"Results saved to {csv_path}")
    
    # Generate visualizations
    generate_visualizations(results_df, experiment_folder)
    
    # Print summary
    print_summary(results_df)
    
    return results_df

def generate_visualizations(results_df, experiment_folder):
    """Generate comparison visualizations."""
    # Set style
    plt.style.use('default')  # Use default style instead of seaborn
    
    # 1. Accuracy Comparison
    plt.figure(figsize=(12, 6))
    # Create bar plot manually since seaborn is causing issues
    x = np.arange(len(results_df['dataset'].unique()))
    width = 0.35
    
    for i, model in enumerate(results_df['model'].unique()):
        model_data = results_df[results_df['model'] == model]
        plt.bar(x + i*width, model_data['accuracy'], width, label=model)
    
    plt.title('Model Comparison: Validation Accuracy by Dataset')
    plt.xlabel('Dataset')
    plt.ylabel('Validation Accuracy')
    plt.xticks(x + width/2, results_df['dataset'].unique(), rotation=45, ha='right')
    plt.legend(title='Model Type')
    plt.tight_layout()
    plt.savefig(os.path.join(experiment_folder, 'accuracy_comparison.png'))
    
    # 2. Average Performance
    avg_performance = results_df.groupby('model')['accuracy'].mean().reset_index()
    plt.figure(figsize=(8, 6))
    plt.bar(avg_performance['model'], avg_performance['accuracy'])
    plt.title('Average Performance Across Datasets')
    plt.xlabel('Model Type')
    plt.ylabel('Average Validation Accuracy')
    plt.tight_layout()
    plt.savefig(os.path.join(experiment_folder, 'average_performance.png'))

def print_summary(results_df):
    """Print a detailed summary of the results."""
    print("\n=== SAFE vs SAFE with GrowingNN Comparison ===")
    print("\nDetailed Results:")
    print(results_df.to_string(index=False))
    
    print("\nAverage Performance:")
    avg_performance = results_df.groupby('model')['accuracy'].agg(['mean', 'std']).round(4)
    print(avg_performance)
    
    print("\nBest Performance by Dataset:")
    best_by_dataset = results_df.loc[results_df.groupby('dataset')['accuracy'].idxmax()]
    print(best_by_dataset[['dataset', 'model', 'accuracy']].to_string(index=False))

if __name__ == '__main__':
    results_df = run_comparison()
