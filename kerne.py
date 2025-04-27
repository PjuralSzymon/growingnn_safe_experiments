"""
Time Series Classification using SAX and Neural Networks

This module implements a time series classification system using Symbolic Aggregate approXimation (SAX)
and neural networks (both standard and GrowingNN). The pipeline consists of:
1. Data loading and normalization
2. SAX transformation
3. Word extraction from symbolic sequences
4. Document vectorization
5. Neural network training (standard or GrowingNN)

Author: Unknown
"""

# Standard library imports
import logging
import os
from typing import List, Tuple, Dict, Union, Any

# Third-party imports - Data processing
import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sktime.datasets import load_UCR_UEA_dataset

# Third-party imports - Deep Learning
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, GlobalAveragePooling1D, Dense, Dropout
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam

# Local imports
import growingnn as gnn

# Configure logging with clear stage separators
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class Config:
    """Configuration class to hold all hyperparameters and settings."""
    
    class SAX:
        """SAX transformation settings"""
        ALPHABET_SIZE = 8
        WORD_LENGTH = 4
    
    class NeuralNetwork:
        """Neural network settings"""
        EMBEDDING_DIM = 4
        DROPOUT_RATE = 0.1
        L2_LAMBDA = 1e-4
        EPOCHS = 20
        BATCH_SIZE = 32
        HIDDEN_UNITS = 32
    
    class GrowingNN:
        """GrowingNN specific settings"""
        SIMULATION_TIME = 60
        SIMULATION_EPOCHS = 20
        SIMULATION_SET_SIZE = 20
        BATCH_SIZE = 12
        LEARNING_RATE = 0.03
        LR_DECAY = 0.8
    
    class Training:
        """General training settings"""
        VALIDATION_SPLIT = 0.2
        RANDOM_SEED = 42

def log_stage(stage_title):
    logging.info("\n" + "="*80)
    logging.info("---- %s ----", stage_title)
    logging.info("="*80)

# ------------------------ Step 1: Load and Normalize Time Series ------------------------
def load_and_normalize_time_series(dataset_name, split='train'):
    log_stage(f"Loading and Normalizing Data (Dataset: {dataset_name}, Split: {split})")
    X, y = load_UCR_UEA_dataset(name=dataset_name, split=split, return_X_y=True)
    
    # Extract the single column of time series from the DataFrame
    ts = X.iloc[:, 0].tolist()
    X_norm = []
    for i, series in enumerate(ts):
        # Convert series to float and flatten it
        series = np.array(series, dtype=float).ravel()
        # Compute z-normalization: (value - mean) / std
        mu = np.mean(series)
        sigma = np.std(series)
        if sigma == 0:
            normalized = series - mu  # avoid division by zero
        else:
            normalized = (series - mu) / sigma
        X_norm.append(normalized)
    X_norm = np.array(X_norm)
    logging.info("Normalized %d time series, each of length %d", X_norm.shape[0], X_norm.shape[1])
    logging.debug("Example time series (first sample): %s", X_norm[0])
    return X_norm, y

# ------------------------ Custom SAX Discretization (Without PAA) ------------------------
def custom_sax_transform(data, alphabet_size=8):
    """
    Converts each normalized time series into a symbolic representation.
    Each time point is discretized independently using breakpoints derived from the standard normal distribution.
    The output has the same shape as the input.
    """
    log_stage(f"Custom SAX Discretization (Alphabet Size: {alphabet_size})")
    n_samples, n_points = data.shape
    # Compute breakpoints for the given alphabet size using quantiles of N(0,1)
    # Exclude -infty and +infty: we get alphabet_size - 1 breakpoints.
    breakpoints = norm.ppf(np.linspace(0, 1, alphabet_size + 1)[1:-1])
    logging.info("Computed breakpoints: %s", breakpoints)
    
    # Initialize the symbolic representation array (as integer symbols)
    data_sax = np.empty_like(data, dtype=int)
    for i in range(n_samples):
        for j in range(n_points):
            # Find the index where the normalized value fits in the breakpoints.
            # np.searchsorted returns the insertion index such that the sequence remains sorted.
            symbol = np.searchsorted(breakpoints, data[i, j])
            data_sax[i, j] = symbol
    logging.info("Custom SAX transformation complete. Shape: %s", data_sax.shape)
    logging.debug("Example symbolic series (first sample): %s", data_sax[0])
    return data_sax

# ------------------------ Step 3: Extract Words from Symbolic Time Series ------------------------
def extract_words(data_sax, word_length=4):
    log_stage(f"Extracting Non-Overlapping Words (Word Length: {word_length})")
    documents = []
    for series in data_sax:
        # Map numeric symbols to letters (0 -> 'a', 1 -> 'b', etc.)
        symbols = [chr(97 + int(s)) for s in series]  # 97 is ASCII for 'a'
        series_str = ''.join(symbols)
        # Extract non-overlapping words by stepping with word_length
        words = [series_str[i:i+word_length] for i in range(0, len(series_str), word_length)
                 if len(series_str[i:i+word_length]) == word_length]
        documents.append(' '.join(words))  # Form a document (string) of words separated by spaces
    logging.info("Extracted words for %d documents.", len(documents))
    logging.debug("Example document (first sample): %s", documents[0])
    return documents

# ------------------------ Step 4: Vectorize Documents ------------------------
def vectorize_documents(documents, max_words=None):
    log_stage("Vectorizing Documents")
    tokenizer = Tokenizer(num_words=max_words, lower=True, split=' ')
    tokenizer.fit_on_texts(documents)
    sequences = tokenizer.texts_to_sequences(documents)
    word_index = tokenizer.word_index
    logging.info("Tokenizer found %d unique tokens.", len(word_index))
    
    logging.debug("Sequences. (first sample): %s", sequences[0])
    first_key = next(iter(word_index))  # Get the first key safely
    logging.debug("Word_index (first sample): %s -> %s", first_key, word_index[first_key])
    return sequences, word_index, tokenizer

# ------------------------ Step 5: Prepare Neural Network Input ------------------------
def prepare_nn_input(sequences, max_length):
    log_stage(f"Padding Sequences to Length {max_length}")
    padded_sequences = pad_sequences(sequences, maxlen=max_length, padding='post', truncating='post')
    logging.info("NN input shape after padding: %s", padded_sequences.shape)
    return padded_sequences

# ------------------------ Step 6: Build and Train Neural Network ------------------------
def build_and_train_nn(x_train, y_train, x_val, y_val, vocab_size, input_length, num_classes, 
                         embedding_dim=4, dropout_rate=0.1, l2_lambda=1e-4, epochs=100):
    log_stage("Building and Training Neural Network Model")
    model = Sequential([
        # Embedding layer converts token indices into dense vectors
        Embedding(input_dim=vocab_size + 1, output_dim=embedding_dim, input_length=input_length),
        GlobalAveragePooling1D(),  # Aggregates the embeddings over the time dimension
        Dense(32, activation='relu', kernel_regularizer=l2(l2_lambda)),
        Dropout(dropout_rate),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer=Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.summary(print_fn=lambda x: logging.info(x))
    
    logging.info("Starting model training for %d epochs...", epochs)
    history = model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=epochs, batch_size=32)
    logging.info("Model training complete.")
    return model, history


import growingnn as gnn
import os
def train_growingnn(x_train, y_train, x_val, y_val, labels, input_size, hidden_size, output_size, 
                     epochs=10, generations=5, model_name="growingnn_model", is_cnn=False):
    """
    Train a GrowingNN model with proper shape handling
    """
    logging.info("Training an Adaptive Neural Network with GrowingNN")
    logging.info(f"Original shapes - x_train: {x_train.shape}, y_train: {y_train.shape}")
    
    # Create save path
    save_path = os.path.join(os.getcwd(), "growingnn_models")
    os.makedirs(save_path, exist_ok=True)
    
    # 1. Prepare input data - transpose to (features, samples)
    x_train = x_train.T
    x_val = x_val.T
    
    # 2. Prepare labels - ensure they are integers
    y_train = y_train.astype(int)
    y_val = y_val.astype(int)
    
    logging.info(f"Prepared shapes:")
    logging.info(f"- x_train: {x_train.shape}")
    logging.info(f"- y_train: {y_train.shape}")
    logging.info(f"- x_val: {x_val.shape}")
    logging.info(f"- y_val: {y_val.shape}")
    
    try:
        trained_model = gnn.trainer.train(
            path="./model_output/",
            x_train=x_train, 
            y_train=y_train,    
            x_test=x_val, 
            y_test=y_val,          
            labels=list(range(output_size)),                        
            input_size=x_train.shape[0],                
            hidden_size=hidden_size,              
            output_size=output_size,              
            model_name=model_name,                
            epochs=epochs, 
            generations=generations, 
            input_shape=None,             
            kernel_size=None,   
            deepth=2,  
            batch_size=12, 
            simulation_set_size=20, 
            simulation_alg=gnn.montecarlo_alg, 
            sim_set_generator=gnn.create_simulation_set_SAMLE,
            simulation_scheduler=gnn.SimulationScheduler(
                gnn.SimulationScheduler.PROGRESS_CHECK, 
                simulation_time=60, 
                simulation_epochs=20
            ),
            lr_scheduler=gnn.LearningRateScheduler(
                gnn.LearningRateScheduler.PROGRESIVE, 
                0.03, 0.8
            ),
            loss_function=gnn.Loss.multiclass_cross_entropy,
            activation_fun=gnn.Activations.Sigmoid,
            optimizer=gnn.optimizers.SGDOptimizer()
        )

        logging.info("GrowingNN model training complete")
        
        # Evaluate the model
        accuracy = trained_model.evaluate(x_val, y_val)
        logging.info(f"GrowingNN Validation Accuracy: {accuracy:.4f}")
        return {'accuracy': accuracy}
        
    except Exception as e:
        logging.error(f"GrowingNN training failed: {str(e)}")
        raise

def train_time_series_classifier(
    dataset_name: str = 'ArrowHead',
    use_growing_nn: bool = False,
    word_length: int = Config.SAX.WORD_LENGTH,
    embedding_dim: int = Config.NeuralNetwork.EMBEDDING_DIM,
    epochs: int = Config.NeuralNetwork.EPOCHS,
    generations: int = 5
) -> Dict[str, float]:
    """Train a time series classifier using either standard NN or GrowingNN.
    
    Args:
        dataset_name: Name of the UCR/UEA dataset to use
        use_growing_nn: Whether to use GrowingNN (True) or standard NN (False)
        word_length: Length of words for SAX transformation
        embedding_dim: Dimension of word embeddings (for standard NN)
        epochs: Number of training epochs
        generations: Number of generations for GrowingNN

    Returns:
        Dictionary containing training metrics
    """
    # Stage 1: Load and normalize data
    X_train_raw, y_train = load_and_normalize_time_series(dataset_name, split='train')
    X_test_raw, y_test = load_and_normalize_time_series(dataset_name, split='test')
    
    # Combine data for corpus building
    X_all = np.concatenate((X_train_raw, X_test_raw), axis=0)
    y_all = np.concatenate((y_train, y_test), axis=0)
    logging.info(f"Combined dataset shape: {X_all.shape}")
    
    # Stage 2: SAX transformation and word extraction
    sax = SAXTransformer()
    data_sax = sax.transform(X_all)
    documents = sax.extract_words(data_sax, word_length=word_length)
    
    # Stage 3: Vectorize documents
    tokenizer = Tokenizer(lower=True, split=' ')
    tokenizer.fit_on_texts(documents)
    sequences = tokenizer.texts_to_sequences(documents)
    word_index = tokenizer.word_index
    
    # Prepare neural network input
    max_seq_length = max(len(seq) for seq in sequences)
    x_data = pad_sequences(sequences, maxlen=max_seq_length, padding='post', truncating='post')
    
    # Encode labels
    label_encoder = LabelEncoder()
    y_data = label_encoder.fit_transform(y_all)
    num_classes = len(np.unique(y_data))
    
    # Split data
    x_train, x_val, y_train, y_val = train_test_split(
        x_data, y_data,
        test_size=Config.Training.VALIDATION_SPLIT,
        random_state=Config.Training.RANDOM_SEED
    )
    
    logging.info(f"Training data shape: x->{x_train.shape} y->{y_train.shape}")
    logging.info(f"Validation data shape: x->{x_val.shape} y->{y_val.shape}")
    
    if use_growing_nn:
        # Stage 7: Train Using GrowingNN first
        try:
            # Prepare data for GrowingNN
            input_size = x_train.shape[1]  # Number of features
            hidden_size = input_size  # Use same size as input
            output_size = num_classes  # Number of classes
            
            logging.info(f"GrowingNN parameters:")
            logging.info(f"- Input size: {input_size}")
            logging.info(f"- Hidden size: {hidden_size}")
            logging.info(f"- Output size: {output_size}")
            
            # Train GrowingNN model
            result = train_growingnn(
                x_train=x_train,
                y_train=y_train,
                x_val=x_val,
                y_val=y_val,
                labels=list(label_encoder.classes_),
                input_size=input_size,
                hidden_size=hidden_size,
                output_size=output_size,
                epochs=epochs,
                generations=generations,
                model_name=f"growingnn_{dataset_name}",
                is_cnn=False
            )
            
            # Check if GrowingNN training was successful
            if 'accuracy' not in result or result['accuracy'] is None:
                error_msg = "GrowingNN training failed: No accuracy returned"
                logging.error(error_msg)
                raise Exception(error_msg)
                
            return result
            
        except Exception as e:
            error_msg = f"GrowingNN training failed: {str(e)}"
            logging.error(error_msg)
            raise Exception(error_msg)
    else:
        # Train standard neural network
        classifier = StandardNNClassifier(
            vocab_size=len(word_index),
            input_length=max_seq_length,
            num_classes=num_classes,
            embedding_dim=embedding_dim
        )
        history = classifier.fit(x_train, y_train, validation_data=(x_val, y_val))
        metrics = classifier.evaluate(x_val, y_val)
        return metrics

class SAXTransformer:
    """Handles the SAX transformation and word extraction process."""
    
    def __init__(self, alphabet_size: int = Config.SAX.ALPHABET_SIZE):
        """Initialize the SAX transformer.
        
        Args:
            alphabet_size: Number of symbols in the SAX alphabet
        """
        self.alphabet_size = alphabet_size
        self.breakpoints = self._compute_breakpoints()
        
    def _compute_breakpoints(self) -> np.ndarray:
        """Compute SAX breakpoints using N(0,1) distribution."""
        return norm.ppf(np.linspace(0, 1, self.alphabet_size + 1)[1:-1])
    
    def transform(self, data: np.ndarray) -> np.ndarray:
        """Transform normalized time series into symbolic representation.
        
        Args:
            data: Normalized time series data of shape (n_samples, n_timepoints)
            
        Returns:
            Symbolic representation of the same shape as input
        """
        log_stage(f"SAX Discretization (Alphabet Size: {self.alphabet_size})")
        n_samples, n_points = data.shape
        data_sax = np.empty_like(data, dtype=int)
        
        for i in range(n_samples):
            data_sax[i] = np.searchsorted(self.breakpoints, data[i])
            
        logging.info(f"SAX transformation complete. Shape: {data_sax.shape}")
        return data_sax
    
    def extract_words(self, data_sax: np.ndarray, word_length: int = Config.SAX.WORD_LENGTH) -> List[str]:
        """Extract words from symbolic representation.
        
        Args:
            data_sax: Symbolic representation of time series
            word_length: Length of each word
            
        Returns:
            List of documents, where each document is a space-separated string of words
        """
        log_stage(f"Extracting Words (Length: {word_length})")
        documents = []
        
        for series in data_sax:
            # Convert symbols to letters (0 -> 'a', 1 -> 'b', etc.)
            symbols = [chr(97 + s) for s in series]
            series_str = ''.join(symbols)
            
            # Extract non-overlapping words
            words = [
                series_str[i:i+word_length] 
                for i in range(0, len(series_str), word_length)
                if len(series_str[i:i+word_length]) == word_length
            ]
            documents.append(' '.join(words))
            
        logging.info(f"Extracted words for {len(documents)} documents")
        return documents

class BaseTimeSeriesClassifier:
    """Base class for time series classifiers."""
    
    def __init__(self):
        self.model = None
        self.label_encoder = LabelEncoder()
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        """Train the model."""
        raise NotImplementedError
    
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """Evaluate the model."""
        raise NotImplementedError

class StandardNNClassifier(BaseTimeSeriesClassifier):
    """Standard neural network classifier using word embeddings."""
    
    def __init__(self, 
                 vocab_size: int,
                 input_length: int,
                 num_classes: int,
                 embedding_dim: int = Config.NeuralNetwork.EMBEDDING_DIM):
        super().__init__()
        self.vocab_size = vocab_size
        self.input_length = input_length
        self.num_classes = num_classes
        self.embedding_dim = embedding_dim
        self._build_model()
    
    def _build_model(self):
        """Build the neural network architecture."""
        self.model = Sequential([
            Embedding(
                input_dim=self.vocab_size + 1,
                output_dim=self.embedding_dim,
                input_length=self.input_length
            ),
            GlobalAveragePooling1D(),
            Dense(
                Config.NeuralNetwork.HIDDEN_UNITS,
                activation='relu',
                kernel_regularizer=l2(Config.NeuralNetwork.L2_LAMBDA)
            ),
            Dropout(Config.NeuralNetwork.DROPOUT_RATE),
            Dense(self.num_classes, activation='softmax')
        ])
        self.model.compile(
            optimizer=Adam(),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
    
    def fit(self, X: np.ndarray, y: np.ndarray, validation_data=None):
        """Train the model.
        
        Args:
            X: Input sequences
            y: Target labels
            validation_data: Tuple of (X_val, y_val) for validation
        """
        history = self.model.fit(
            X, y,
            validation_data=validation_data,
            epochs=Config.NeuralNetwork.EPOCHS,
            batch_size=Config.NeuralNetwork.BATCH_SIZE
        )
        return history
    
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """Evaluate the model.
        
        Args:
            X: Input sequences
            y: Target labels
            
        Returns:
            Dictionary containing loss and accuracy metrics
        """
        loss, accuracy = self.model.evaluate(X, y, verbose=0)
        return {
            'loss': loss,
            'accuracy': accuracy
        }

class GrowingNNClassifier(BaseTimeSeriesClassifier):
    """Classifier using the GrowingNN framework."""
    
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
    def fit(self, X: np.ndarray, y: np.ndarray, validation_data=None, generations: int = 5):
        """Train the GrowingNN model.
        
        Args:
            X: Input data
            y: Target labels
            validation_data: Tuple of (X_val, y_val) for validation
            generations: Number of generations for growing
        """
        X_val, y_val = validation_data if validation_data else (None, None)
        
        # Ensure data is in the correct shape
        X = X.reshape(X.shape[0], -1) if len(X.shape) != 2 else X
        if X_val is not None:
            X_val = X_val.reshape(X_val.shape[0], -1) if len(X_val.shape) != 2 else X_val
        
        self.model = gnn.trainer.train(
            path="./model_output/",
            x_train=X,
            y_train=y,
            x_test=X_val,
            y_test=y_val,
            labels=list(range(self.output_size)),
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            output_size=self.output_size,
            epochs=Config.NeuralNetwork.EPOCHS,
            generations=generations,
            batch_size=Config.GrowingNN.BATCH_SIZE,
            simulation_set_size=Config.GrowingNN.SIMULATION_SET_SIZE,
            simulation_alg=gnn.montecarlo_alg,
            sim_set_generator=gnn.create_simulation_set_SAMLE,
            simulation_scheduler=gnn.SimulationScheduler(
                gnn.SimulationScheduler.PROGRESS_CHECK,
                simulation_time=Config.GrowingNN.SIMULATION_TIME,
                simulation_epochs=Config.GrowingNN.SIMULATION_EPOCHS
            ),
            lr_scheduler=gnn.LearningRateScheduler(
                gnn.LearningRateScheduler.PROGRESIVE,
                Config.GrowingNN.LEARNING_RATE,
                Config.GrowingNN.LR_DECAY
            ),
            loss_function=gnn.Loss.multiclass_cross_entropy,
            activation_fun=gnn.Activations.Sigmoid,
            optimizer=gnn.optimizers.SGDOptimizer()
        )
        return self.model
    
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """Evaluate the GrowingNN model.
        
        Args:
            X: Input data
            y: Target labels
            
        Returns:
            Dictionary containing accuracy metric
        """
        X = X.reshape(X.shape[0], -1) if len(X.shape) != 2 else X
        accuracy = self.model.evaluate(X, y)
        return {'accuracy': accuracy}