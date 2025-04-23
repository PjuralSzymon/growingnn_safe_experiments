# Theoretical Background to the Project

This document outlines the theoretical and methodological background for the project, which integrates two contemporary advances in machine learning applied to time series data: **Dynamic Growing and Shrinking Neural Networks using Monte Carlo Tree Search (MCTS)**, and the **SAFE (Simple And Fast segmented word Embedding-based) time series classification method**. These approaches address fundamental challenges in time series modeling, such as handling dynamic data environments, managing network complexity, and effectively extracting patterns from sequential numerical data.

---

## 1️⃣ Dynamic Growing and Shrinking Neural Networks with Monte Carlo Tree Search (MCTS)

### 📌 Background and Motivation

Neural networks typically require a fixed architecture, specified prior to training. The choice of the number of neurons and layers is often based on heuristic guidelines, grid searches, or manual experimentation. However, in dynamic, real-world data environments — such as financial markets, IoT sensor networks, or user activity streams — the optimal model capacity can shift over time.

A static model might either:
- Underfit, lacking the representational capacity to learn new patterns as the data evolves.
- Overfit or waste computational resources by maintaining an unnecessarily large architecture when simpler models would suffice.

To address this, the paper proposes a **dynamic neural network framework**, where the architecture can adapt — **growing and shrinking neurons dynamically during training** — guided by a decision-making mechanism based on **Monte Carlo Tree Search (MCTS)**.

---

### 📌 Methodological Framework

The core of the method involves integrating an MCTS algorithm into the neural network training loop to guide structural modifications:

- **MCTS as an Optimization Heuristic:**  
  MCTS is traditionally used in AI for games (like AlphaGo) to explore large decision trees efficiently by balancing exploration (trying new actions) and exploitation (choosing known good actions). Here, MCTS is repurposed to decide, at each training iteration, whether to add neurons, remove them, or retain the current structure.

- **State Representation:**  
  Each state in the MCTS represents a specific network configuration, defined by the number and arrangement of neurons across layers.

- **Actions:**  
  The possible actions are:
  - **Grow:** Add new neurons to a layer.
  - **Shrink:** Remove existing neurons.
  - **Do nothing:** Keep the architecture unchanged.

- **Simulation and Evaluation:**  
  Each potential architecture is evaluated using a performance metric (like validation loss) to simulate the outcomes of different sequences of actions.

- **Search Tree Update:**  
  The MCTS tree is updated iteratively by selecting actions based on a balance of performance gains (exploitation) and trying alternative architectures (exploration).

---

### 📌 Empirical Results and Advantages

- **Performance Gains:**  
  The adaptive networks consistently outperformed fixed-size networks, particularly in non-stationary and data-limited scenarios.

- **Compact Models:**  
  The method naturally prunes redundant neurons, avoiding unnecessary complexity and reducing computational costs.

- **Generalization:**  
  By maintaining a model size proportional to task difficulty, the adaptive networks demonstrated superior generalization across diverse datasets.

- **Scalability:**  
  The framework is applicable to various neural architectures and learning tasks beyond classification, including regression and anomaly detection in time series.

---

## 2️⃣ SAFE: Simple And Fast segmented word Embedding-based Time Series Classifier

### 📌 Background and Motivation

Time series classification is an increasingly important area in domains ranging from healthcare (ECG analysis) to industry (sensor data) and multimedia (gesture recognition). Among classification strategies, **dictionary-based methods** are particularly effective for handling unequal-length series and complex, high-dimensional data.

Traditional dictionary-based classifiers, like **BOSS** and **WEASEL**, rely on discretizing time series into symbols, forming "words" from subsequences, and using frequency histograms for classification. However, these methods face limitations:
- High-dimensional histograms can become sparse.
- Important contextual information between subsequences is often ignored.

**SAFE** addresses these issues by integrating ideas from **Natural Language Processing (NLP)** — specifically word embeddings — to learn contextualized representations of symbolic subsequences within time series.

---

### 📌 Methodological Framework

The SAFE method involves three primary steps:

#### 1. Symbolic Transformation
The time series data, originally a sequence of continuous numerical values, is transformed into a sequence of discrete symbols using the **Simple Symbolic Aggregate approXimation (SAX)** technique. SAX reduces dimensionality while preserving important shape-based information, mapping subsequences to symbols from a predefined alphabet.

#### 2. Segmentation and Word Formation
Symbolic sequences are partitioned into consecutive "words" — fixed-length sequences of symbols. Each time series is thereby converted into a sequence of words, analogous to a document in NLP.

#### 3. Embedding-Based Classification
The resulting "documents" are fed into a neural network model featuring a **word embedding layer**. This layer maps symbolic words into dense, low-dimensional vectors that capture contextual relationships between different subsequences (i.e., words appearing in similar contexts will have similar embeddings).

A fully connected classification head predicts the time series label based on the learned embeddings.

---

### 📌 Key Contributions and Advantages

- **High Accuracy:**  
  SAFE consistently ranked in the top 5–10% across 30 benchmark time series datasets, outperforming or matching leading dictionary-based and neural classifiers.

- **Compact, Efficient Models:**  
  Despite using neural networks, model sizes remained small (around 20k parameters), making them suitable for edge computing applications.

- **No Dimensionality Reduction Needed:**  
  Unlike other dictionary methods (e.g., BOSS, SAX-VSM), SAFE does not require truncating or downsampling time series, preserving all available information.

- **Contextual Pattern Learning:**  
  The embedding layer captures relationships between subsequences based on their surrounding context, allowing the classifier to discover higher-order temporal patterns often missed by histogram-based methods.

- **Versatility:**  
  Demonstrated excellent performance across diverse data types: ECG signals, image contour data, motion capture, simulated data, and spectral measurements.

---

## 🔗 Synthesis and Integration

Combining these two methods creates a powerful, flexible system for time series analysis:
- **MCTS-Driven Dynamic Architectures** ensure that the model’s complexity adapts to the task's evolving demands, avoiding both underfitting and overfitting.
- **SAFE’s Symbolic Representation and Contextual Pattern Mining** allow complex, high-dimensional time series data to be distilled into symbolic sequences, enabling efficient, interpretable, and accurate classification.

This hybrid framework is especially well-suited to applications involving:
- Streaming or evolving data environments.
- Embedded systems and edge devices.
- Heterogeneous datasets with variable-length or irregular time series.

---

## 📚 References

- Kudlek, M., et al. (2024). *Dynamic Growing and Shrinking of Neural Networks with Monte Carlo Tree Search*. In Lecture Notes in Computer Science, Springer.
- Karwowski, R., et al. (2022). *SAFE: Simple And Fast segmented word Embedding-based neural time series classifier*. Cognitive Systems Research, Elsevier.

---
