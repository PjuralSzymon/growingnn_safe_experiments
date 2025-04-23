import logging
import numpy as np
import pandas as pd
import kerne

# ------------------------ Main Execution ------------------------
if __name__ == '__main__':
    kerne.train_time_series_classifier(useGrowingnn=False,
                                 dataset_name='ArrowHead', 
                                 word_length=4, 
                                 embedding_dim=4, 
                                 epochs=10, 
                                 generations=5,
                                 )