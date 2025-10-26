# Resume Selector with AI Classification

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://www.tensorflow.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0%2B-yellow.svg)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An advanced machine learning solution for automated resume screening and classification using both traditional ML (Naive Bayes) and deep learning (Keras Neural Networks) approaches.

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Model Architecture](#model-architecture)
- [Performance Metrics](#performance-metrics)
- [Data Pipeline](#data-pipeline)
- [Technologies Used](#technologies-used)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

## 🎯 Overview

This project implements an intelligent resume classification system that automatically flags or validates resumes based on their content. The solution leverages natural language processing (NLP) techniques and provides two distinct machine learning approaches:

1. **Naive Bayes Classifier** - Fast, probabilistic approach for baseline performance
2. **Deep Neural Network (Keras)** - Advanced deep learning model for complex pattern recognition

The system is designed to handle real-world resume data with comprehensive preprocessing, feature extraction, and model evaluation pipelines.

## ✨ Features

- **Dual Model Approach**: Compare traditional ML vs. deep learning performance
- **Advanced Text Preprocessing**: 
  - Stop word removal
  - Text normalization
  - Feature extraction using CountVectorizer
- **Comprehensive Visualization**:
  - Word clouds for class analysis
  - Confusion matrices
  - Training history plots
  - Class distribution analysis
- **Production-Ready**:
  - Model persistence (save/load functionality)
  - Detailed performance metrics
  - Classification reports
  - Reproducible pipeline
- **Imbalanced Data Handling**: Strategies for dealing with unbalanced datasets

## 📁 Project Structure

```
Resume-Selector/
│
├── Resume Selector with Naive Bayes_AP23110010253.ipynb  # Main notebook
├── resume.csv                                              # Dataset
├── resume_selector_keras_model.h5                         # Trained Keras model
├── README.md                                              # Project documentation
└── requirements.txt                                       # Python dependencies
```

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager
- Virtual environment (recommended)

### Setup Instructions

1. **Clone the repository**
   ```bash
   git clone https://github.com/Likhith623/Resume-Selector.git
   cd Resume-Selector
   ```

2. **Create a virtual environment** (recommended)
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Download NLTK data**
   ```python
   python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"
   ```

### Requirements

Create a `requirements.txt` file with the following dependencies:

```
pandas>=1.3.0
numpy>=1.21.0
matplotlib>=3.4.0
seaborn>=0.11.0
nltk>=3.6.0
gensim>=4.0.0
wordcloud>=1.8.0
scikit-learn>=1.0.0
tensorflow>=2.8.0
jupyterthemes>=0.20.0
jupyter>=1.0.0
```

## 💻 Usage

### Running the Notebook

1. **Start Jupyter Notebook**
   ```bash
   jupyter notebook
   ```

2. **Open the main notebook**
   - Navigate to `Resume Selector with Naive Bayes_AP23110010253.ipynb`
   - Run cells sequentially from top to bottom

### Using the Trained Model

```python
# Load the saved Keras model
from tensorflow import keras
model = keras.models.load_model('resume_selector_keras_model.h5')

# Prepare new resume data
# (Apply same preprocessing as training data)
new_resume_vectorized = vectorizer.transform([preprocessed_text])
new_resume_dense = new_resume_vectorized.toarray()

# Make prediction
prediction = model.predict(new_resume_dense)
result = "Flagged" if prediction[0][0] > 0.5 else "Not Flagged"
print(f"Resume Status: {result}")
```

## 🏗️ Model Architecture

### Naive Bayes Classifier

- **Algorithm**: MultinomialNB
- **Feature Extraction**: CountVectorizer
- **Training Time**: ~1-2 seconds
- **Best Use Case**: Quick baseline, interpretable results

### Keras Neural Network

```
Model Architecture:
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
dense_1 (Dense)              (None, 512)               N * 512   
dropout_1 (Dropout)          (None, 512)               0         
dense_2 (Dense)              (None, 256)               131,328   
dropout_2 (Dropout)          (None, 256)               0         
dense_3 (Dense)              (None, 128)               32,896    
dropout_3 (Dropout)          (None, 128)               0         
dense_4 (Dense)              (None, 64)                8,256     
dropout_4 (Dropout)          (None, 64)                0         
dense_5 (Dense)              (None, 1)                 65        
=================================================================
```

**Key Features**:
- **Activation**: ReLU (hidden layers), Sigmoid (output)
- **Regularization**: Dropout (0.5 and 0.3)
- **Optimizer**: Adam (learning_rate=0.001)
- **Loss Function**: Binary Crossentropy
- **Training**: 20 epochs, batch_size=32

## 📊 Performance Metrics

### Evaluation Metrics

Both models are evaluated using:

- **Accuracy**: Overall correctness
- **Precision**: Positive predictive value
- **Recall**: Sensitivity/True positive rate
- **F1-Score**: Harmonic mean of precision and recall
- **Confusion Matrix**: Detailed classification breakdown

### Visualization

The project includes comprehensive visualizations:

1. **Class Distribution**: Understanding data imbalance
2. **Word Clouds**: Visual representation of important terms
3. **Confusion Matrices**: Model performance breakdown
4. **Training History**: Loss and accuracy curves (Keras model)

## 🔄 Data Pipeline

### 1. Data Loading
```python
resume_df = pd.read_csv("resume.csv", encoding="latin-1")
```

### 2. Preprocessing
- Remove special characters (`\r`)
- Tokenization
- Stop word removal
- Filter words with length > 2

### 3. Feature Extraction
- CountVectorizer for text-to-numeric conversion
- Sparse matrix representation

### 4. Model Training
- Train-test split (70-30 or configurable)
- Model fitting
- Validation

### 5. Evaluation
- Predictions on test set
- Performance metrics calculation
- Visualization

## 🛠️ Technologies Used

| Category | Technologies |
|----------|-------------|
| **Language** | Python 3.8+ |
| **ML Frameworks** | scikit-learn, TensorFlow/Keras |
| **NLP** | NLTK, Gensim |
| **Data Processing** | Pandas, NumPy |
| **Visualization** | Matplotlib, Seaborn, WordCloud |
| **Development** | Jupyter Notebook |

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### Development Guidelines

- Follow PEP 8 style guide
- Add docstrings to functions
- Include unit tests for new features
- Update documentation as needed

## 📝 Future Enhancements

- [ ] Implement BERT/Transformer-based models
- [ ] Add REST API for model serving
- [ ] Create web interface for resume upload
- [ ] Implement multi-class classification (job categories)
- [ ] Add explainability features (LIME/SHAP)
- [ ] Optimize model hyperparameters
- [ ] Add data augmentation techniques
- [ ] Implement cross-validation
- [ ] Create Docker containerization
- [ ] Add CI/CD pipeline

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👤 Contact

**Likhith**

- GitHub: [@Likhith623](https://github.com/Likhith623)
- Project Link: [https://github.com/Likhith623/Resume-Selector](https://github.com/Likhith623/Resume-Selector)

## 🙏 Acknowledgments

- NLTK team for natural language processing tools
- TensorFlow/Keras team for deep learning framework
- scikit-learn community for machine learning utilities
- All contributors and supporters of this project

---

**Note**: This project is for educational and research purposes. Ensure compliance with data privacy regulations when using with real resume data.

## 📚 References

- [NLTK Documentation](https://www.nltk.org/)
- [TensorFlow Documentation](https://www.tensorflow.org/)
- [scikit-learn Documentation](https://scikit-learn.org/)
- [Keras Guide](https://keras.io/)

---

<div align="center">
Made with ❤️ by Likhith
</div>
