# Trendaizer

Trendaizer is a project that leverages deep learning to forecast trends in the stock market, helping users make informed decisions. By utilizing historical stock market data, we train a model that can anticipate future trends with the aid of Long Short Term Memory (LSTM) networks.


Trendaizer aims to predict weekly stock market trends, enabling investors and traders to gain insights into potential market movements. The application uses historical stock data to train an LSTM model, which forecasts the probability of upward or downward trends.

## Arquitechture 

The core of the Trendaizer project focuses on a Long Short Term Memory (LSTM) model for stock market trend prediction. The model architecture is made up of the following components:

1. **LSTM Layers**:
 - The network uses two LSTM layers with 50 units each. These layers are responsible for capturing temporal dependencies in historical price data.
 - Each LSTM layer is equipped with 'relu' activation function to handle nonlinearity in the data.

2. **Regularization**:
 - An L2 regularization (`kernel_regularizer=l2(0.0005)`) is used in each LSTM layer and in the final dense layer to prevent overfitting of the model.
 - Dropout layers (`Dropout(0.02)`) are added after each LSTM layer to improve model generalization.

3. **Dense Layer**:
 - A dense layer (`Dense(1, activation='sigmoid')`) at the end of the network provides the final output, which is the prediction of the weekly trend. The 'sigmoid' activation function converts the output into a probability between 0 and 1.

4. **Compilation**:
 - The model is compiled with the 'adam' optimizer, an 'mse' (mean squared error) loss function, and the accuracy metric to evaluate the model's performance during training.

## Process 

1. **Data Collection**: 
   - Historical stock data is downloaded using the `yfinance` library.
   - Data is collected for selected stocks (e.g., AAPL, MSFT, TSLA).

2. **Data Preprocessing**:
   - Calculate weekly returns and trends.
   - Generate weekly features (log returns of the last 4 weeks).
   - Scale the data using `StandardScaler`.

3. **Model Training**:
   - Train an LSTM model with the processed data.
   - The model includes regularization and dropout layers to prevent overfitting.
   - The model is trained for 150 epochs with a batch size of 32.

4. **Prediction**:
   - The trained model predicts the probability of weekly trends.
   - The predictions are compared with actual trends to evaluate accuracy.

![texto](https://cdn.discordapp.com/attachments/699128484097818684/1261090905956290721/Screenshot_2024-07-04_212821.png?ex=6691b1ed&is=6690606d&hm=053aa3db8dd0b3d2b1124bf3729c81aa6d9b4d065a91781b5ab0fff860cc4ce6&)

## Project state 

The project is currently decently efecctive. The LSTM model has been trained and tested with historical data, and the initial version of the Streamlit application is operational. Future enhancements include improving model accuracy, adding more features, and enhancing the web interface.

## Special Thanks 

A special thanks to Samsung Innovation Campus for the opportunity to acquire all the knowledge provided to open the doors to this field of artificial intelligence.

And to our teacher Giancarlo Colasante who accompanied us throughout this teaching career and also made it possible for us to take full advantage of his valuable knowledge and experiences.

## 🛠️ Installation 

### 💻 Local Installation

To set up the project locally, follow these steps:

1. Clone the repository:
    ```bash
    git https://github.com/repositoriosHackaton/CodeChamers.git
    cd trendaizer
    ```

2. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3. Run the Streamlit application:
    ```bash
    streamlit run app.py
    ```

## Use 

1. Open the Streamlit application in your web browser.
2. Select a stock from the dropdown menu.
3. View the predicted trend for the selected stock.
4. Check the graphical representation of closing prices and moving averages.

### Resources that help us:
[Theorycal explanation of Prediction of stock return by LSTM neural network](https://www.tandfonline.com/doi/full/10.1080/08839514.2022.2151159)

[Common mistakes on the use of LSTM](https://www.youtube.com/watch?v=Vfx1L2jh2Ng)

