# 🤖 Machine Learning Tasks

> Hands-on implementations of core machine learning algorithms and techniques — from classical models to neural networks — built to understand how things work under the hood, not just how to call `.fit()`.

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## What This Is

A task-based ML lab: each script/notebook tackles one concept end-to-end on real datasets (Titanic, Housing, Social Network Ads, MNIST, Dogs vs Cats).

## Topics Covered

**Classical ML**
- Decision Trees & ensemble methods (voting, bagging)
- Cross-validation & model evaluation (confusion matrix)
- Data cleaning, feature scaling & normalization
- Handling imbalanced datasets

**Neural Networks**
- ANN: binary classification, regression, batch normalization
- Weight initialization (He) & Keras Tuner hyperparameter search
- CNN: MNIST digit classifier, Dogs vs Cats image classifier

## Structure

```
├── *.py                  # Standalone algorithm implementations
├── *.ipynb               # Experiment notebooks (ANN, CNN)
├── *.csv                 # Practice datasets
└── README.md
```

## Tech

Python · scikit-learn · TensorFlow/Keras · pandas · NumPy · imbalanced-learn

## Running a Task

```bash
pip install scikit-learn pandas numpy tensorflow
python decisiontree.py
# or open any .ipynb in Jupyter / VS Code
```

## License

MIT — see [LICENSE](LICENSE).
