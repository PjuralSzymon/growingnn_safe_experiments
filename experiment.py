import os
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import kerne
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Define datasets for experiments (More datasets can be added here)
DATASETS = [
    'ArrowHead', 'Beef', 'Coffee', 'ECG200', 'FaceFour', 'GunPoint', 'Lightning2', 'OSULeaf',
    'SonyAIBORobotSurface1', 'SwedishLeaf', 'ToeSegmentation1', 'Wafer', 'WordSynonyms', 'Yoga'
]

# Hyperparameters
WORD_LENGTH = 4
EMBEDDING_DIM = 4
EPOCHS = 10
GENERATIONS = 5

# Create a unique experiment folder
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
experiment_folder = f"experiments/exp_{timestamp}"
os.makedirs(experiment_folder, exist_ok=True)

# Store results
results = []

# ------------------------ Run Experiments ------------------------
if __name__ == '__main__':
    for dataset in DATASETS:
        for use_gnn in [False, True]:  # Run both TensorFlow and GrowingNN models
            model_type = "GrowingNN" if use_gnn else "TensorFlow"
            logging.info(f"Running experiment: Dataset={dataset}, Model={model_type}")

            try:
                # Train model
                result = kerne.train_time_series_classifier(
                    dataset_name=dataset, 
                    word_length=WORD_LENGTH, 
                    embedding_dim=EMBEDDING_DIM, 
                    epochs=EPOCHS, 
                    useGrowingnn=use_gnn
                )
                
                # Append results
                result['dataset'] = dataset
                result['model'] = model_type
                results.append(result)

            except Exception as e:
                logging.error(f"Training failed for {dataset} ({model_type}): {e}")
                results.append({'dataset': dataset, 'model': model_type, 'train_accuracy': None, 'val_accuracy': None})

    # Convert results to a DataFrame
    results_df = pd.DataFrame(results)

    # Save results to CSV
    csv_path = os.path.join(experiment_folder, "results.csv")
    results_df.to_csv(csv_path, index=False)
    logging.info(f"Results saved to {csv_path}")

    # ------------------------ Generate Summary Graphs ------------------------

    # Accuracy Comparison Graph
    plt.figure(figsize=(10, 5))
    for model in ["TensorFlow", "GrowingNN"]:
        subset = results_df[results_df["model"] == model]
        plt.plot(subset["dataset"], subset["val_accuracy"], marker='o', label=f"{model} Val Accuracy")

    plt.xticks(rotation=45, ha='right')
    plt.xlabel("Dataset")
    plt.ylabel("Validation Accuracy")
    plt.title("Model Comparison: Validation Accuracy")
    plt.legend()
    plt.tight_layout()

    # Save graph
    accuracy_plot_path = os.path.join(experiment_folder, "accuracy_comparison.png")
    plt.savefig(accuracy_plot_path)
    logging.info(f"Accuracy graph saved: {accuracy_plot_path}")

    print("\n=== Experiment Results Summary ===")
    print(results_df.to_string(index=False))
