import os
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def generate_real_time_visualizations(results_df, graphs_folder, dataset, parameter_grid):
    """Generate real-time visualizations after each experiment."""
    # Filter results for the current dataset
    dataset_results = results_df[results_df['dataset'] == dataset]
    
    # Skip if no results yet
    if len(dataset_results) == 0:
        return
    
    # Set style
    plt.style.use('default')
    
    # 1. Parameter Importance Analysis - Bar Charts
    for param in parameter_grid.keys():
        plt.figure(figsize=(10, 6))
        
        # Group by parameter value and calculate mean accuracy
        param_importance = dataset_results.groupby(param)['accuracy'].mean().sort_values(ascending=False)
        
        # Create bar plot
        plt.bar(range(len(param_importance)), param_importance.values)
        plt.xticks(range(len(param_importance)), param_importance.index, rotation=45)
        
        plt.title(f'Parameter Importance: {param} for {dataset}')
        plt.xlabel(param)
        plt.ylabel('Mean Accuracy')
        plt.tight_layout()
        
        # Save figure
        plt.savefig(os.path.join(graphs_folder, f'{param}_importance.png'))
        plt.close()
    
    # 2. Parameter Correlation Heatmap
    if len(dataset_results) > 1:
        plt.figure(figsize=(12, 10))
        
        # Select numeric columns for correlation
        numeric_cols = ['accuracy'] + [col for col in parameter_grid.keys() 
                                      if isinstance(dataset_results[col].iloc[0], (int, float))]
        
        if len(numeric_cols) > 1:
            # Calculate correlation matrix
            corr_matrix = dataset_results[numeric_cols].corr()
            
            # Create heatmap
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f')
            plt.title(f'Parameter Correlation for {dataset}')
            plt.tight_layout()
            
            # Save figure
            plt.savefig(os.path.join(graphs_folder, 'parameter_correlation.png'))
            plt.close()
    
    # 3. Accuracy Distribution
    plt.figure(figsize=(10, 6))
    
    # Create histogram
    plt.hist(dataset_results['accuracy'], bins=10, alpha=0.7)
    plt.axvline(dataset_results['accuracy'].mean(), color='r', linestyle='dashed', linewidth=2, label=f'Mean: {dataset_results["accuracy"].mean():.4f}')
    plt.axvline(dataset_results['accuracy'].max(), color='g', linestyle='dashed', linewidth=2, label=f'Max: {dataset_results["accuracy"].max():.4f}')
    
    plt.title(f'Accuracy Distribution for {dataset}')
    plt.xlabel('Accuracy')
    plt.ylabel('Frequency')
    plt.legend()
    plt.tight_layout()
    
    # Save figure
    plt.savefig(os.path.join(graphs_folder, 'accuracy_distribution.png'))
    plt.close()
    
    # 4. Parameter vs Accuracy Scatter Plots
    for param in parameter_grid.keys():
        plt.figure(figsize=(10, 6))
        
        # Create scatter plot
        plt.scatter(dataset_results[param], dataset_results['accuracy'], alpha=0.7)
        
        # Add trend line if numeric
        if isinstance(dataset_results[param].iloc[0], (int, float)):
            z = np.polyfit(dataset_results[param], dataset_results['accuracy'], 1)
            p = np.poly1d(z)
            plt.plot(dataset_results[param], p(dataset_results[param]), "r--", alpha=0.8)
        
        plt.title(f'{param} vs Accuracy for {dataset}')
        plt.xlabel(param)
        plt.ylabel('Accuracy')
        plt.tight_layout()
        
        # Save figure
        plt.savefig(os.path.join(graphs_folder, f'{param}_vs_accuracy.png'))
        plt.close()
    
    # 5. Top 5 Parameter Combinations
    if len(dataset_results) >= 5:
        plt.figure(figsize=(12, 8))
        plt.axis('off')
        
        # Get top 5 parameter combinations
        top_5 = dataset_results.nlargest(5, 'accuracy')
        
        # Create table data
        table_data = [['Parameter', 'Value']]
        for _, row in top_5.iterrows():
            for param in parameter_grid.keys():
                table_data.append([param, row[param]])
            table_data.append(['Accuracy', f"{row['accuracy']:.4f}"])
            table_data.append(['', ''])  # Empty row for separation
        
        # Create table
        table = plt.table(cellText=table_data, loc='center', cellLoc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 1.5)
        
        plt.title(f'Top 5 Parameter Combinations for {dataset}')
        plt.tight_layout()
        
        # Save figure
        plt.savefig(os.path.join(graphs_folder, 'top_5_parameters.png'))
        plt.close()

def print_current_best_configs(best_configs):
    """Print the current best configurations for each dataset."""
    print("\n=== Current Best Configurations ===")
    for dataset, config in best_configs.items():
        if config['params'] is not None:
            print(f"\nDataset: {dataset}")
            print(f"Best accuracy: {config['accuracy']:.4f}")
            print("Best parameters:")
            for param_name, param_value in config['params'].items():
                if param_name != 'accuracy':
                    print(f"  {param_name}: {param_value}")

def generate_visualizations(results_df, experiment_folder, parameter_grid):
    """Generate visualizations for parameter analysis."""
    # Set style
    plt.style.use('default')
    
    # 1. Parameter Importance Analysis
    for dataset in results_df['dataset'].unique():
        dataset_results = results_df[results_df['dataset'] == dataset]
        
        # Filter out rows with errors
        valid_results = dataset_results[dataset_results['error'].isna()]
        
        if len(valid_results) == 0:
            logging.warning(f"No valid results for dataset {dataset}")
            continue
        
        # Create a figure for each parameter
        for param in parameter_grid.keys():
            plt.figure(figsize=(10, 6))
            
            # Group by parameter value and calculate mean accuracy
            param_importance = valid_results.groupby(param)['accuracy'].mean().sort_values(ascending=False)
            
            # Create bar plot
            plt.bar(range(len(param_importance)), param_importance.values)
            plt.xticks(range(len(param_importance)), param_importance.index, rotation=45)
            
            plt.title(f'Parameter Importance: {param} for {dataset}')
            plt.xlabel(param)
            plt.ylabel('Mean Accuracy')
            plt.tight_layout()
            
            # Save figure
            plt.savefig(os.path.join(experiment_folder, f'{dataset}_{param}_importance.png'))
            plt.close()
    
    # 2. Best Parameter Combinations
    for dataset in results_df['dataset'].unique():
        dataset_results = results_df[results_df['dataset'] == dataset]
        
        # Filter out rows with errors
        valid_results = dataset_results[dataset_results['error'].isna()]
        
        if len(valid_results) == 0:
            logging.warning(f"No valid results for dataset {dataset}")
            continue
        
        # Get top 5 parameter combinations
        top_5 = valid_results.nlargest(5, 'accuracy')
        
        # Create a table visualization
        plt.figure(figsize=(12, 8))
        plt.axis('off')
        
        # Create table data
        table_data = [['Parameter', 'Value']]
        for _, row in top_5.iterrows():
            for param in parameter_grid.keys():
                table_data.append([param, row[param]])
            table_data.append(['Accuracy', f"{row['accuracy']:.4f}"])
            table_data.append(['', ''])  # Empty row for separation
        
        # Create table
        table = plt.table(cellText=table_data, loc='center', cellLoc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 1.5)
        
        plt.title(f'Top 5 Parameter Combinations for {dataset}')
        plt.tight_layout()
        
        # Save figure
        plt.savefig(os.path.join(experiment_folder, f'{dataset}_top_5_parameters.png'))
        plt.close()
    
    # 3. Dataset Comparison
    plt.figure(figsize=(12, 8))
    
    # Calculate mean accuracy for each dataset
    dataset_accuracy = results_df.groupby('dataset')['accuracy'].mean().sort_values(ascending=False)
    
    # Create bar plot
    plt.bar(range(len(dataset_accuracy)), dataset_accuracy.values)
    plt.xticks(range(len(dataset_accuracy)), dataset_accuracy.index, rotation=45)
    
    plt.title('Mean Accuracy by Dataset')
    plt.xlabel('Dataset')
    plt.ylabel('Mean Accuracy')
    plt.tight_layout()
    
    # Save figure
    plt.savefig(os.path.join(experiment_folder, 'dataset_comparison.png'))
    plt.close()
    
    # 4. Parameter Correlation Heatmap
    plt.figure(figsize=(12, 10))
    
    # Select numeric columns for correlation
    numeric_cols = ['accuracy'] + [col for col in parameter_grid.keys() 
                                  if isinstance(results_df[col].iloc[0], (int, float))]
    
    if len(numeric_cols) > 1:
        # Calculate correlation matrix
        corr_matrix = results_df[numeric_cols].corr()
        
        # Create heatmap
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f')
        plt.title('Parameter Correlation')
        plt.tight_layout()
        
        # Save figure
        plt.savefig(os.path.join(experiment_folder, 'parameter_correlation.png'))
        plt.close() 