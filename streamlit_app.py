import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import yfinance as yf
import ta
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

from sklearn.metrics import mean_squared_error, mean_absolute_error
import math

# Configuração do Streamlit
st.set_page_config(page_title="Forecast Financeiro", layout="wide", page_icon="📈")

# Estilo dos gráficos
sns.set(style="darkgrid")

# Título do dashboard
st.title("📊 Dashboard de Forecast para BTC e AAPL")
st.markdown("""
Previsão de preços usando modelos de deep learning (LSTM, GRU, SimpleRNN) com dados históricos do Yahoo Finance.
""")

# Sidebar - Controles do usuário
st.sidebar.header("Configurações")

# Funções do modelo (adaptadas do script original)
@st.cache_data
def load_data(ticker, start_date, end_date):
    """Carrega dados históricos do Yahoo Finance"""
    data = yf.download(ticker, start=start_date, end=end_date)
    data.rename(columns={
        'Open': 'open',
        'High': 'high',
        'Low': 'low',
        'Close': 'close',
        'Adj Close': 'adj_close',
        'Volume': 'volume'
    }, inplace=True)
    return data

@st.cache_data
def calculate_technical_indicators(data):
    """Calcula indicadores técnicos"""
    close = data['close'].astype(float).squeeze()
    
    data['SMA_20'] = ta.trend.sma_indicator(close, window=20)
    data['EMA_20'] = ta.trend.ema_indicator(close, window=20)
    data['RSI'] = ta.momentum.rsi(close, window=14)
    
    macd = ta.trend.MACD(close=close)
    data['MACD'] = macd.macd()
    data['MACD_Signal'] = macd.macd_signal()
    data['MACD_Diff'] = macd.macd_diff()
    
    data.dropna(inplace=True)
    return data

@st.cache_data
def normalize_data(data):
    """Normaliza os dados"""
    scaler = MinMaxScaler()
    features = data[['close', 'SMA_20', 'EMA_20', 'RSI', 'MACD', 'MACD_Signal', 'MACD_Diff']]
    normalized = scaler.fit_transform(features)
    return normalized, scaler

@st.cache_resource
def build_lstm_model(input_shape):
    """Constrói modelo LSTM"""
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(50),
        Dropout(0.2),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

def create_sequences(data, look_back=30):
    """Cria sequências para o modelo"""
    X, y = [], []
    for i in range(len(data) - look_back):
        X.append(data[i:(i + look_back)])
        y.append(data[i + look_back, 0])  # Prever apenas o preço de fechamento
    return np.array(X), np.array(y)

# Interface do usuário
ticker = st.sidebar.selectbox("Selecione o ativo:", ['BTC-USD', 'AAPL'])
model_type = st.sidebar.selectbox("Selecione o modelo:", ['LSTM', 'GRU', 'SimpleRNN'])
look_back = st.sidebar.slider("Janela de look-back (dias):", 15, 60, 30)
forecast_days = st.sidebar.slider("Dias para prever:", 1, 30, 14)

# Carregar dados
st.subheader(f"Dados históricos para {ticker}")
end_date = datetime.now().strftime('%Y-%m-%d')
start_date = '2014-01-01'

with st.spinner('Carregando dados históricos...'):
    data = load_data(ticker, start_date, end_date)
    data = calculate_technical_indicators(data)
    
    # Mostrar dados brutos
    if st.checkbox("Mostrar dados brutos"):
        st.dataframe(data.tail(10))

# Gráfico de preços
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(data.index, data['close'], label='Preço de Fechamento', color='blue')
ax.set_title(f"Histórico de Preços - {ticker}")
ax.set_xlabel("Data")
ax.set_ylabel("Preço (USD)")
ax.legend()
st.pyplot(fig)

# Treinar modelo
if st.button("Treinar Modelo e Gerar Previsões"):
    with st.spinner('Treinando modelo...'):
        # Normalizar dados
        normalized_data, scaler = normalize_data(data)
        
        # Criar sequências
        X, y = create_sequences(normalized_data, look_back)
        
        # Dividir em treino/teste
        train_size = int(len(X) * 0.8)
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]
        
        # Construir e treinar modelo
        model = build_lstm_model((X_train.shape[1], X_train.shape[2]))
        
        history = model.fit(
            X_train, y_train,
            epochs=50,
            batch_size=32,
            validation_split=0.1,
            callbacks=[EarlyStopping(patience=5)],
            verbose=0
        )
        
        # Fazer previsões
        predictions = model.predict(X_test)
        
        # Inverter a normalização
        dummy = np.zeros((len(predictions), normalized_data.shape[1]))
        dummy[:, 0] = predictions.flatten()
        predictions = scaler.inverse_transform(dummy)[:, 0]
        
        dummy[:, 0] = y_test.flatten()
        y_test = scaler.inverse_transform(dummy)[:, 0]
        
        # Criar DataFrame com resultados
        test_dates = data.index[-len(y_test):]
        results = pd.DataFrame({
            'Date': test_dates,
            'Actual': y_test,
            'Predicted': predictions
        }).set_index('Date')
        
        # Calcular métricas
        rmse = math.sqrt(mean_squared_error(y_test, predictions))
        mae = mean_absolute_error(y_test, predictions)
        mape = np.mean(np.abs((y_test - predictions) / y_test)) * 100
        
        # Mostrar métricas
        col1, col2, col3 = st.columns(3)
        col1.metric("RMSE", f"{rmse:.2f}")
        col2.metric("MAE", f"{mae:.2f}")
        col3.metric("MAPE", f"{mape:.2f}%")
        
        # Gráfico de previsões
        fig2, ax2 = plt.subplots(figsize=(12, 6))
        ax2.plot(results.index, results['Actual'], label='Valor Real', color='blue')
        ax2.plot(results.index, results['Predicted'], label='Previsão', color='red', linestyle='--')
        ax2.set_title(f"Previsões vs Valores Reais - {ticker}")
        ax2.set_xlabel("Data")
        ax2.set_ylabel("Preço (USD)")
        ax2.legend()
        st.pyplot(fig2)
        
        # Previsão para os próximos dias
        st.subheader(f"Previsão para os próximos {forecast_days} dias")
        
        last_sequence = normalized_data[-look_back:]
        forecast = []
        
        for _ in range(forecast_days):
            # Fazer previsão
            next_pred = model.predict(last_sequence.reshape(1, look_back, -1))
            
            # Atualizar sequência
            new_row = np.append(next_pred, last_sequence[-1, 1:])
            last_sequence = np.vstack([last_sequence[1:], new_row])
            
            # Armazenar previsão
            forecast.append(next_pred[0, 0])
        
        # Inverter normalização
        dummy = np.zeros((len(forecast), normalized_data.shape[1]))
        dummy[:, 0] = np.array(forecast)
        forecast_prices = scaler.inverse_transform(dummy)[:, 0]
        
        # Criar datas futuras
        last_date = data.index[-1]
        future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=forecast_days)
        
        # Mostrar previsão
        forecast_df = pd.DataFrame({
            'Date': future_dates,
            'Forecast': forecast_prices
        }).set_index('Date')
        
        fig3, ax3 = plt.subplots(figsize=(12, 6))
        ax3.plot(data.index[-100:], data['close'][-100:], label='Histórico', color='blue')
        ax3.plot(forecast_df.index, forecast_df['Forecast'], label='Previsão', color='green', marker='o')
        ax3.set_title(f"Previsão para os próximos {forecast_days} dias - {ticker}")
        ax3.set_xlabel("Data")
        ax3.set_ylabel("Preço (USD)")
        ax3.legend()
        st.pyplot(fig3)
        
        st.dataframe(forecast_df.style.format({'Forecast': '{:.2f}'}))

# Rodapé
st.sidebar.markdown("---")
st.sidebar.markdown("""
**Instruções:**
1. Selecione um ativo e modelo
2. Ajuste os parâmetros
3. Clique em "Treinar Modelo"
""")