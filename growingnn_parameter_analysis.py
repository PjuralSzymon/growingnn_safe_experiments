import os
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from itertools import product
from datetime import datetime, timedelta
import kerne
from sklearn.model_selection import train_test_split
import traceback
import time
import seaborn as sns
from visualization_utils import (
    generate_real_time_visualizations,
    generate_visualizations,
    print_current_best_configs
)

# ===== EXPERIMENT MODE CONTROL =====
# Set to None for new experiment, or provide folder path to continue from previous run
# Example: "growingnn_analysis/exp_20240315_123456"
CONTINUE_FROM = "growingnn_analysis\exp_20250525_123657"  
# ===================================

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Define datasets for experiments (using a subset for faster analysis)
DATASETS = [
    'Coffee',         # 2 classes, 56 samples, shape: (56, 286) - 16,016 total points
    'Beef',           # 5 classes, 60 samples, shape: (60, 470) - 28,200 total points
    'OliveOil',       # 4 classes, 60 samples, shape: (60, 570) - 34,200 total points
    'ECG200',         # 2 classes, 200 samples, shape: (200, 96) - 19,200 total points
    'GunPoint',       # 2 classes, 200 samples, shape: (200, 150) - 30,000 total points
    'Lightning2',     # 2 classes, 121 samples, shape: (121, 637) - 77,077 total points
    'ArrowHead',      # 3 classes, 211 samples, shape: (211, 251) - 52,961 total points
    'SonyAIBORobotSurface1', # 2 classes, 621 samples, shape: (621, 70) - 43,470 total points
    'SonyAIBORobotSurface2', # 2 classes, 980 samples, shape: (980, 65) - 63,700 total points
    'Adiac',          # 37 classes, 781 samples, shape: (781, 176) - 137,456 total points
    'ECGFiveDays',    # 2 classes, 884 samples, shape: (884, 136) - 120,224 total points
    'ItalyPowerDemand', # 2 classes, 1096 samples, shape: (1096, 24) - 26,304 total points
    'TwoLeadECG',     # 2 classes, 1162 samples, shape: (1162, 82) - 95,284 total points
    'MoteStrain',     # 2 classes, 1272 samples, shape: (1272, 84) - 106,848 total points
    'SwedishLeaf',    # 15 classes, 1125 samples, shape: (1125, 128) - 144,000 total points
    'ElectricDevices' # 7 classes, 16637 samples, shape: (16637, 96) - 1,597,152 total points
]

# Define parameter ranges to explore
PARAMETER_GRID = {
    'epochs': [10],
    'generations': [5],
    'hidden_size': [32, 64],
    'batch_size': [32, 64],
    'simulation_set_size': [50, 100, 200],
    'simulation_time': [30, 60, 500],
    'simulation_epochs': [10, 20],
    'simulation_scheduler_type': ['constant', 'progress_check']
}

def format_time_remaining(seconds):
    """Format time remaining in a human-readable format."""
    days = seconds // (24 * 3600)
    seconds = seconds % (24 * 3600)
    hours = seconds // 3600
    seconds %= 3600
    minutes = seconds // 60
    seconds %= 60
    
    parts = []
    if days > 0:
        parts.append(f"{int(days)} day{'s' if days != 1 else ''}")
    if hours > 0:
        parts.append(f"{int(hours)} hour{'s' if hours != 1 else ''}")
    if minutes > 0:
        parts.append(f"{int(minutes)} minute{'s' if minutes != 1 else ''}")
    if seconds > 0 or not parts:
        parts.append(f"{int(seconds)} second{'s' if seconds != 1 else ''}")
    
    return " ".join(parts)

def load_previous_results(experiment_folder):
    """Load results from a previous experiment."""
    csv_path = os.path.join(experiment_folder, "parameter_analysis_results.csv")
    if os.path.exists(csv_path):
        return pd.read_csv(csv_path)
    return None

def is_experiment_completed(previous_results, dataset, params):
    """Check if a specific experiment has already been completed."""
    if previous_results is None:
        return False
    
    # Create a mask for the current dataset
    dataset_mask = previous_results['dataset'] == dataset
    
    # For each parameter, check if it matches
    for param_name, param_value in params.items():
        dataset_mask &= (previous_results[param_name] == param_value)
    
    # Check if we have any matching results
    return dataset_mask.any()

def run_parameter_analysis():
    """Run parameter analysis for GrowingNN algorithm."""
    # Create or use experiment folder based on CONTINUE_FROM
    if CONTINUE_FROM:
        experiment_folder = CONTINUE_FROM
        logging.info(f"Continuing from previous experiment: {experiment_folder}")
        previous_results = load_previous_results(experiment_folder)
        if previous_results is None:
            logging.warning(f"No previous results found in {experiment_folder}, starting new experiment")
            previous_results = pd.DataFrame()
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        experiment_folder = f"growingnn_analysis/exp_{timestamp}"
        os.makedirs(experiment_folder, exist_ok=True)
        previous_results = pd.DataFrame()
        logging.info(f"Starting new experiment: {experiment_folder}")
    
    # Create a folder for real-time graphs
    graphs_folder = os.path.join(experiment_folder, "real_time_graphs")
    os.makedirs(graphs_folder, exist_ok=True)
    
    # Create a common folder for aggregated visualizations
    common_graphs_folder = os.path.join(graphs_folder, "common")
    os.makedirs(common_graphs_folder, exist_ok=True)
    
    # Store results
    results = previous_results.to_dict('records') if not previous_results.empty else []
    
    # Generate all parameter combinations
    param_names = list(PARAMETER_GRID.keys())
    param_values = list(PARAMETER_GRID.values())
    param_combinations = list(product(*param_values))
    
    # Calculate total number of experiments
    total_experiments = len(DATASETS) * len(param_combinations)
    completed_experiments = len(results)
    start_time = time.time()
    
    # Log experiment setup
    logging.info(f"Starting parameter analysis with {len(DATASETS)} datasets and {len(param_combinations)} parameter combinations")
    logging.info(f"Total experiments to run: {total_experiments}")
    logging.info(f"Already completed: {completed_experiments}")
    logging.info(f"Remaining experiments: {total_experiments - completed_experiments}")
    logging.info(f"Parameter grid: {PARAMETER_GRID}")
    
    # Create a CSV file to save results incrementally
    csv_path = os.path.join(experiment_folder, "parameter_analysis_results.csv")
    if not os.path.exists(csv_path):
        pd.DataFrame().to_csv(csv_path, index=False)
    
    # Track best configurations for each dataset
    best_configs = {dataset: {'accuracy': 0.0, 'params': None} for dataset in DATASETS}
    
    # Update best configs from previous results
    if not previous_results.empty:
        for dataset in DATASETS:
            dataset_results = previous_results[previous_results['dataset'] == dataset]
            if not dataset_results.empty:
                best_idx = dataset_results['accuracy'].idxmax()
                best_configs[dataset]['accuracy'] = dataset_results.loc[best_idx, 'accuracy']
                best_configs[dataset]['params'] = {param: dataset_results.loc[best_idx, param] 
                                                 for param in PARAMETER_GRID.keys()}
    
    # Run experiments for each dataset and parameter combination
    for dataset_idx, dataset in enumerate(DATASETS):
        logging.info(f"Analyzing parameters for dataset: {dataset} ({dataset_idx+1}/{len(DATASETS)})")
        
        # Create a folder for this dataset's graphs
        dataset_graphs_folder = os.path.join(graphs_folder, dataset)
        os.makedirs(dataset_graphs_folder, exist_ok=True)
        
        # Use kerne's function to load and preprocess data
        X_train_raw, y_train = kerne.load_and_normalize_time_series(dataset, split='train')
        X_test_raw, y_test = kerne.load_and_normalize_time_series(dataset, split='test')
        
        # Combine data for preprocessing
        X_all = np.concatenate((X_train_raw, X_test_raw), axis=0)
        y_all = np.concatenate((y_train, y_test), axis=0)
        
        # Apply SAX transformation using kerne's SAXTransformer
        sax = kerne.SAXTransformer()
        data_sax = sax.transform(X_all)
        documents = sax.extract_words(data_sax, word_length=kerne.Config.SAX.WORD_LENGTH)
        
        # Vectorize documents
        tokenizer = kerne.Tokenizer(lower=True, split=' ')
        tokenizer.fit_on_texts(documents)
        sequences = tokenizer.texts_to_sequences(documents)
        
        # Prepare neural network input
        max_seq_length = max(len(seq) for seq in sequences)
        x_data = kerne.pad_sequences(sequences, maxlen=max_seq_length, padding='post', truncating='post')
        
        # Encode labels - ensure they are integers
        label_encoder = kerne.LabelEncoder()
        y_data = label_encoder.fit_transform(y_all)
        
        # Ensure labels are integers
        unique_labels = np.unique(y_data)
        label_map = {label: i for i, label in enumerate(unique_labels)}
        y_data = np.array([label_map[label] for label in y_data])
        
        # Split data
        x_train, x_val, y_train, y_val = train_test_split(
            x_data, y_data,
            test_size=kerne.Config.Training.VALIDATION_SPLIT,
            random_state=kerne.Config.Training.RANDOM_SEED
        )
        
        # Prepare GrowingNN parameters
        input_size = x_train.shape[1]
        output_size = len(np.unique(y_data))
        
        # Run experiments for each parameter combination
        for i, param_values in enumerate(param_combinations):
            params = dict(zip(param_names, param_values))
            
            # Skip if this experiment was already completed
            if is_experiment_completed(previous_results, dataset, params):
                logging.info(f"Skipping already completed experiment for {dataset} with params {params}")
                continue
            
            # Update progress
            completed_experiments += 1
            elapsed_time = time.time() - start_time
            avg_time_per_exp = elapsed_time / completed_experiments
            estimated_remaining = avg_time_per_exp * (total_experiments - completed_experiments)
            
            # Calculate progress percentages
            progress_percent = (completed_experiments / total_experiments) * 100
            dataset_progress_percent = ((i + 1) / len(param_combinations)) * 100
            
            # Calculate completion time
            completion_time = datetime.now() + timedelta(seconds=estimated_remaining)
            
            # Log detailed progress information
            logging.info(f"Progress: {progress_percent:.2f}% ({completed_experiments}/{total_experiments})")
            logging.info(f"Dataset progress: {dataset_progress_percent:.2f}% ({i+1}/{len(param_combinations)}) for {dataset}")
            logging.info(f"Elapsed time: {format_time_remaining(elapsed_time)}")
            logging.info(f"Average time per experiment: {format_time_remaining(avg_time_per_exp)}")
            logging.info(f"Estimated time remaining: {format_time_remaining(estimated_remaining)}")
            logging.info(f"Estimated completion time: {completion_time.strftime('%d.%m.%Y %H:%M:%S')}")
            logging.info(f"Testing parameter combination {i+1}/{len(param_combinations)} for dataset {dataset}")
            
            # Create save path
            save_path = os.path.join(os.getcwd(), "growingnn_models")
            os.makedirs(save_path, exist_ok=True)
            
            # Set model parameters
            input_shape = (input_size,)
            kernel_size = None
            depth = 2
            
            # Log training configuration
            logging.info(f"GrowingNN training configuration:")
            for param_name, param_value in params.items():
                logging.info(f"- {param_name}: {param_value}")
            
            # Train the model using kerne's train_growingnn function with all parameters
            result = kerne.train_growingnn(
                x_train=x_train,
                y_train=y_train,
                x_val=x_val,
                y_val=y_val,
                labels=list(range(output_size)),
                input_size=input_size,
                hidden_size=params['hidden_size'],
                output_size=output_size,
                epochs=params['epochs'],
                generations=params['generations'],
                batch_size=params['batch_size'],
                simulation_set_size=params['simulation_set_size'],
                simulation_time=params['simulation_time'],
                simulation_epochs=params['simulation_epochs'],
                simulation_scheduler_type=params['simulation_scheduler_type'],
                model_name=f"growingnn_{dataset}_{i}",
                is_cnn=False
            )
            
            # Store results
            result_dict = {
                'dataset': dataset,
                'accuracy': result.get('accuracy', 0.0),
                'error': None
            }
            
            # Add all parameters to the result
            for param_name, param_value in params.items():
                result_dict[param_name] = param_value
            
            results.append(result_dict)
            
            # Update best configuration for this dataset
            if result.get('accuracy', 0.0) > best_configs[dataset]['accuracy']:
                best_configs[dataset]['accuracy'] = result.get('accuracy', 0.0)
                best_configs[dataset]['params'] = params.copy()
                best_configs[dataset]['params']['accuracy'] = result.get('accuracy', 0.0)
                logging.info(f"New best accuracy for {dataset}: {best_configs[dataset]['accuracy']:.4f}")
            
            # Save results incrementally after each experiment
            results_df = pd.DataFrame(results)
            results_df.to_csv(csv_path, index=False)
            
            # Generate real-time visualizations after each experiment
            generate_real_time_visualizations(results_df, dataset_graphs_folder, dataset, PARAMETER_GRID)
            
            # Generate aggregated visualizations for all datasets
            generate_real_time_visualizations(results_df, common_graphs_folder, "all_datasets", PARAMETER_GRID)
            
            # Print current best configurations
            print_current_best_configs(best_configs)
        
        # Log completion of dataset
        logging.info(f"Completed all experiments for dataset {dataset}")
    
    # Log completion of all experiments
    total_time = time.time() - start_time
    logging.info(f"Completed all {total_experiments} experiments in {format_time_remaining(total_time)}")
    
    # Convert results to DataFrame
    results_df = pd.DataFrame(results)
    
    # Generate final visualizations
    generate_visualizations(results_df, experiment_folder, PARAMETER_GRID)
    
    # Print summary
    print_summary(results_df)
    
    return results_df

def print_summary(results_df):
    """Print a summary of the parameter analysis results."""
    print("\n=== GrowingNN Parameter Analysis Summary ===")
    
    for dataset in results_df['dataset'].unique():
        print(f"\nDataset: {dataset}")
        
        # Filter results for this dataset
        dataset_results = results_df[results_df['dataset'] == dataset]
        
        # Count successful and failed experiments
        successful = dataset_results[dataset_results['error'].isna()]
        failed = dataset_results[dataset_results['error'].notna()]
        
        print(f"Total experiments: {len(dataset_results)}")
        print(f"Successful: {len(successful)}")
        print(f"Failed: {len(failed)}")
        
        if len(successful) > 0:
            # Find best parameters
            best_result = successful.loc[successful['accuracy'].idxmax()]
            print(f"\nBest accuracy: {best_result['accuracy']:.4f}")
            print("Best parameters:")
            for param in PARAMETER_GRID.keys():
                print(f"  {param}: {best_result[param]}")
        
        # Parameter importance
        if len(successful) > 0:
            print("\nParameter importance (correlation with accuracy):")
            for param in PARAMETER_GRID.keys():
                correlation = successful[param].corr(successful['accuracy'])
                print(f"  {param}: {correlation:.4f}")

if __name__ == '__main__':
    try:
        results_df = run_parameter_analysis()
    except Exception as e:
        print(f"Error: {str(e)}")
        print("Full stack trace:")
        traceback.print_exc() 