# Stock Price Prediction

This project focuses on predicting stock prices using LSTM (Long Short-Term Memory) neural networks. It uses historical stock data to train the model and provides a simple interface for users to predict and visualize stock prices.

## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Model Training](#model-training)
- [Streamlit Application](#streamlit-application)
- [Contributing](#contributing)
- [License](#license)

## Installation

1. **Clone the repository:**
    ```sh
    git clone https://github.com/your-username/stock-price-prediction.git
    cd stock-price-prediction
    ```

2. **Create and activate a virtual environment:**
    ```sh
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. **Install the required packages:**
    ```sh
    pip install -r requirements.txt
    ```

## Usage

### Model Training

1. **Run the Jupyter notebook for training the model:**
    Open `StockPricePrediction.ipynb` in Jupyter Notebook or Google Colab and run the cells step-by-step to train the model and save it as `stockModel.h5`.

### Streamlit Application

1. **Run the Streamlit application:**
    ```sh
    streamlit run app.py
    ```

2. **Interact with the application:**
    - Enter the stock symbol (e.g., `GOOG` for Google).
    - View the historical stock data and moving averages.
    - See the comparison of the original and predicted stock prices.

## Project Structure


## Model Training

The model is built using a Sequential LSTM network with the following architecture:
- 4 LSTM layers with Dropout
- A Dense output layer

The model is trained on Google's stock data from Yahoo Finance, with an 80-20 train-test split.

### Training Parameters

- **Optimizer:** Adam
- **Loss Function:** Mean Squared Error
- **Epochs:** 50
- **Batch Size:** 32

## Streamlit Application

The Streamlit app provides a user-friendly interface for predicting stock prices. Key features include:
- Display of historical stock data.
- Visualization of moving averages (50, 100, 200 days).
- Comparison of original vs. predicted stock prices.

## Contributing

Contributions are welcome! Please fork the repository and create a pull request with your changes.

## License

This project is licensed under the MIT License.

