import pandas as pd
import numpy as np
import joblib
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import TimeSeriesSplit

def build_model(y):
    tscv = TimeSeriesSplit(n_splits=7)
    mse_scores = []

    for train_index, test_index in tscv.split(y):
        train, test = y.iloc[train_index], y.iloc[test_index]
        
        # Ejemplo de hiperparámetros fijos, considera optimizar con GridSearchCV o RandomizedSearchCV
        p, d, q, P, D, Q, s = (0, 1, 1, 0, 0, 1, 3)
        
        model = SARIMAX(train, order=(p, d, q), seasonal_order=(P, D, Q, s))
        model_fit = model.fit(disp=False)

        # Hacer predicciones
        predictions = model_fit.forecast(steps=len(test))
        
        # Calcular el MSE y almacenarlo
        mse = mean_squared_error(test, predictions)
        mse_scores.append(mse)

    average_mse = np.mean(mse_scores)
    print(f'MSE promedio: {average_mse}')

    return model_fit

def save_model(model, file_path):
    joblib.dump(model, file_path)

def load_model(file_path):
    return joblib.load(file_path)

def make_predictions(model, steps):
    predictions = model.forecast(steps=steps)
    return predictions

if __name__ == '__main__':
    model_file_path = 'Model/model_01.pkl'
    future_steps = 3

    # Cargar datos
    historical_data = pd.read_csv(r'Datawarehouse\pizzas_normales_sd.csv')
    historical_data['fecha'] = pd.to_datetime(historical_data['fecha'])  # Convertir a datetime
    historical_data = historical_data.set_index('fecha')  # Establecer la fecha como índice
    historical_data.index.freq = 'MS'  # 'MS' significa inicio de mes

    # Preparar los datos para el modelo SARIMA
    y = historical_data['unidades_total']

    # Construir y entrenar el modelo
    model_fit = build_model(y)

    # Guardar el modelo entrenado
    save_model(model_fit, model_file_path)

    # Cargar el modelo guardado
    loaded_model = load_model(model_file_path)

    # Hacer predicciones futuras
    future_predictions = make_predictions(loaded_model, future_steps)
    print(f'Predicciones futuras: {future_predictions}')