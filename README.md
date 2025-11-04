# 🔮 Unown

**Unown** is a machine learning-powered Pokémon legendary status classifier that predicts whether a Pokémon is legendary based on its stats, abilities, and other attributes.

## ✨ Features

- **ML-Powered Predictions**: Random Forest classifier trained on comprehensive Pokémon dataset
- **Interactive Web App**: Beautiful Streamlit interface for predictions
- **Feature Engineering**: Advanced domain-informed features for better accuracy
- **Two Prediction Modes**:
  - Predict existing Pokémon from the dataset
  - Create custom Pokémon and predict their legendary status
- **Model Insights**: View feature importance and model performance metrics

## 🚀 Installation

### Prerequisites

- Python 3.12 or higher
- [uv](https://github.com/astral-sh/uv) package manager (recommended) or pip

### Setup

1. Clone the repository:
```bash
git clone https://github.com/shravanasati/unown.git
cd unown
```

2. Install dependencies:

Using uv (recommended):
```bash
uv sync
```

Or using pip:
```bash
pip install .
```

## 📖 Usage

### Running the Streamlit Web App

Launch the interactive web application:

```bash
streamlit run streamlit_app.py
```

The app will open in your browser at `http://localhost:8501`. From there you can:
- Select existing Pokémon to see predictions
- Create custom Pokémon with your own stats and abilities
- View model performance metrics and feature importance
- Explore engineered features used in predictions

### Running the CLI

For a simple command-line interface:

```bash
python main.py
```

### Downloading Fresh Dataset

To download the latest Pokémon data using pypokedex:

```bash
python dataset_download.py
```

This will fetch Pokémon data (up to #1025) and save it to `pokemon_data.csv`.

## 🤖 Model Information

The classifier uses a **Random Forest** model with optimized hyperparameters, trained on features including:

- **Ability legendary rates**: How often abilities appear on legendary Pokémon
- **Type combination rates**: Legendary rates for specific type combinations
- **Base stats**: HP, Attack, Defense, Special Attack, Special Defense, Speed
- **Derived stats**: Base total, stat means, physical/special totals
- **Base experience**: Experience points gained from defeating the Pokémon
- **Generation**: Which generation the Pokémon belongs to

The model achieves strong performance with:
- Optimized decision threshold for balanced precision/recall
- Feature importance analysis to understand predictions
- Cross-validation for robust evaluation

## 🛠️ Development

### Feature Engineering

The `FeatureBuilder` class in `feature_engineering.py` creates domain-informed features:
- Calculates legendary rates for abilities, types, and generations
- Normalizes numerical features (stats, height, weight)
- Handles missing values and rare categories

### Model Training

Open `model_training.ipynb` in Jupyter to:
- Explore the dataset
- Train and tune the Random Forest model
- Evaluate performance with various metrics
- Export model artifacts

## 📄 License

This project is licensed under the BSD 3-Clause License - see the [LICENSE.txt](LICENSE.txt) file for details.

## 🙏 Acknowledgments

- Pokémon data sourced from [pypokedex](https://github.com/arnavb/pypokedex) and [PokéAPI](https://pokeapi.co/)
- Built with [Streamlit](https://streamlit.io/), [scikit-learn](https://scikit-learn.org/), and [pandas](https://pandas.pydata.org/)
