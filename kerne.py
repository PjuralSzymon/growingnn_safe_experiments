import logging
import numpy as np
import pandas as pd
from sktime.datasets import load_UCR_UEA_dataset
from scipy.stats import norm
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, GlobalAveragePooling1D, Dense, Dropout
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Configure logging with clear stage separators
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

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
    x_train = x_train.transpose()
    x_val = x_val.transpose()

    logging.info("Training an Adaptive Neural Network with GrowingNN")
    
    save_path = os.path.join(os.getcwd(), "growingnn_models")
    os.makedirs(save_path, exist_ok=True)
    
    # Fix input shape: if using CNN mode, the shape would be (height, width, channels)
    input_shape = (input_size, input_size, 1) if is_cnn else None

    logging.info("labels: %s", len(labels))
    logging.info("input_size: %s", input_size)
    logging.info("hidden_size: %s", hidden_size)
    logging.info("output_size: %s", output_size)
    print("is_cnn: ", is_cnn)
    trained_model = gnn.trainer.train(
        path="./model_output/",
        x_train=x_train, 
        y_train=y_train,    
        x_test=x_val, 
        y_test=y_val,          
        labels=labels,                        
        input_size=input_size,                
        hidden_size=hidden_size,              
        output_size=output_size,              
        #path=save_path,                       
        model_name=model_name,                
        epochs=epochs, generations=generations, 
        # Specify CNN or Dense mode
        input_shape=None,             
        kernel_size=None,   
        deepth=None,  
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

    logging.info("GrowingNN model training complete and saved at %s", save_path)
    return trained_model

def train_time_series_classifier(useGrowingnn=False,
                                 dataset_name='ArrowHead', 
                                 word_length=4, 
                                 embedding_dim=4, 
                                 epochs=100, 
                                 generations=5,
                                 ):
    """
    Trains a time-series classification model on a given dataset.

    Parameters:
    - dataset_name (str): The name of the UCR/UEA time-series dataset to load.
    - word_length (int): The fixed length for extracting non-overlapping words from the SAX-transformed series.
    - embedding_dim (int): The dimensionality of the embedding layer in the neural network.
    - epochs (int): The number of training epochs for the model.
    - useGrowingnn (bool): If True, trains the model using the GrowingNN framework instead of a standard neural network.

    Returns:
    - Trained model (either a standard NN or a GrowingNN model).
    """
    
    # Stage 1: Load and Normalize Data
    X_train_raw, y_train = load_and_normalize_time_series(dataset_name, split='train')
    X_test_raw, y_test = load_and_normalize_time_series(dataset_name, split='test')
    
    # Combine training and testing data to build the corpus
    X_all = np.concatenate((X_train_raw, X_test_raw), axis=0)
    labels_all = np.concatenate((y_train, y_test), axis=0)
    logging.info("Combined dataset shape: %s", X_all.shape)

    # Stage 2: Custom SAX Discretization (without PAA)
    data_sax = custom_sax_transform(X_all, alphabet_size=8)
    
    # Stage 3: Extract Non-Overlapping Words from the Symbolic Series
    documents = extract_words(data_sax, word_length=word_length)
    
    # Stage 4: Vectorize the Documents (build the corpus)
    sequences, word_index, tokenizer = vectorize_documents(documents)
    max_seq_length = max(len(seq) for seq in sequences)
    logging.info("Maximum sequence length (words per document): %d", max_seq_length)
    x_data = prepare_nn_input(sequences, max_seq_length)
    
    # Encode labels into integers
    label_encoder = LabelEncoder()
    y_data = label_encoder.fit_transform(labels_all)
    num_classes = len(np.unique(y_data))
    logging.info("Number of classes: %d", num_classes)
    
    # Stage 5: Split the Data into Training and Validation Sets
    x_train, x_val, y_train_enc, y_val_enc = train_test_split(x_data, y_data, test_size=0.2, random_state=42)
    logging.info("Training data shape: x->%s y->%s | %s", x_train.shape, y_train_enc.shape, type(x_train))
    logging.info("Validation data shape: x->%s y->%s | %s", x_val.shape, y_val_enc.shape, type(x_val))
    logging.info("y_train_enc: %s", y_train_enc[0:10])
    
    if not useGrowingnn:
        # Stage 6: Build and Train the Neural Network Classifier
        vocab_size = len(word_index)
        nn_model, history = build_and_train_nn(
            x_train, y_train_enc, x_val, y_val_enc, 
            vocab_size, 
            max_seq_length, 
            num_classes,
            embedding_dim=embedding_dim, 
            epochs=epochs * generations
        )

        # Evaluate the model
        train_loss, train_accuracy = nn_model.evaluate(x_train, y_train_enc, verbose=0)
        val_loss, val_accuracy = nn_model.evaluate(x_val, y_val_enc, verbose=0)

        return {
            'train_accuracy': train_accuracy,
            'train_loss': train_loss,
            'val_accuracy': val_accuracy,
            'val_loss': val_loss
        }
    
    else:
        # Stage 7: Train Using GrowingNN
        growingnn_model = train_growingnn(
            x_train, y_train_enc, x_val, y_val_enc,
            labels=list(label_encoder.classes_),
            input_size=x_train.shape[1],
            hidden_size=x_train.shape[1], 
            output_size=num_classes,
            epochs=epochs, 
            generations=generations,
            is_cnn=False
        )

        # Extract GrowingNN evaluation metrics (assuming it has an evaluate function or similar)
        train_accuracy = growingnn_model.evaluate(x_train.transpose(), y_train_enc)  # Modify if needed
        val_accuracy = growingnn_model.evaluate(x_val.transpose(), y_val_enc)  # Modify if needed

        return {
            'train_accuracy': train_accuracy,
            'val_accuracy': val_accuracy
        }