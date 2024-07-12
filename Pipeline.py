import pandas as pd
import numpy as np
import joblib
import hashlib
import warnings
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import TimeSeriesSplit

# Suprimir advertencias específicas
warnings.filterwarnings("ignore", category=UserWarning, module="statsmodels")

# Función para cargar datos desde un archivo CSV
def load_data(file_path):
    try:
        df = pd.read_csv(file_path)  # Leer archivo CSV
        df['date'] = pd.to_datetime(df['date'])  # Convertir columna de fechas
        df = df.set_index('date')  # Establecer columna de fechas como índice
        df.index.freq = 'MS'  # Configurar frecuencia del índice
        return df['value']  # Devolver la serie temporal
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

# Función para construir y entrenar el modelo SARIMA
def build_model(y):
    try:
        tscv = TimeSeriesSplit(n_splits=7)  # Dividir la serie temporal en 7 partes
        mse_scores = []

        # Entrenar y validar el modelo con validación cruzada
        for train_index, test_index in tscv.split(y):
            train, test = y.iloc[train_index], y.iloc[test_index]

            # Configuración del modelo SARIMA
            p, d, q, P, D, Q, s = (0, 1, 1, 0, 0, 1, 3)
            model = SARIMAX(train, order=(p, d, q), seasonal_order=(P, D, Q, s))
            model_fit = model.fit(disp=False)

            # Hacer predicciones
            predictions = model_fit.forecast(steps=len(test))
            
            # Calcular el MSE y almacenarlo
            mse = mean_squared_error(test, predictions)
            mse_scores.append(mse)

        # Entrenar el modelo final con todos los datos
        model = SARIMAX(y, order=(p, d, q), seasonal_order=(P, D, Q, s))
        model_fit = model.fit(disp=False)

        average_mse = np.mean(mse_scores)  # MSE promedio de la validación cruzada
        print(f'MSE promedio: {average_mse}')

        mse = mse_scores[-1]  # MSE de la última iteración de validación
        print(f'MSE: {mse}')

        return model_fit, average_mse, mse
    except Exception as e:
        print(f"Error building model: {e}")
        return None

# Función para guardar el modelo entrenado
def save_model(model, model_id):
    try:
        model_hash = generate_number_from_word(model_id)  # Generar un hash a partir del ID del modelo
        filename = f"Model_output/Model_{model_hash}.pkl"  # Nombre del archivo para guardar el modelo
        joblib.dump(model, filename)  # Guardar el modelo con joblib
        return model_hash
    except Exception as e:
        print(f"Error saving model: {e}")
        return None

# Función para cargar un modelo guardado
def load_model(short_timestamp):
    try:
        filename = f"Model_output/Model_{short_timestamp}.pkl"  # Nombre del archivo del modelo guardado
        return joblib.load(filename)  # Cargar el modelo con joblib
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

# Función para hacer predicciones con el modelo cargado
def make_predictions(model, steps):
    try:
        predictions = model.get_forecast(steps=steps)  # Hacer predicciones
        pred_values = predictions.predicted_mean  # Valores predichos
        confidence_intervals = predictions.conf_int()  # Intervalos de confianza
        return pred_values, confidence_intervals
    except Exception as e:
        print(f"Error making predictions: {e}")
        return None, None
    
# Función para visualizar las predicciones y los datos históricos
def visualize_predictions(historical_data, pred_values, confidence_intervals, model_id):
    try:
        # Concatenar los datos históricos con las predicciones para visualización
        predicted_data = pd.concat([historical_data, pred_values])

        # Configurar el gráfico
        plt.figure(figsize=(10, 6))
        plt.plot(predicted_data, label='Predicción y Datos Históricos')
        plt.fill_between(confidence_intervals.index, 
                         confidence_intervals.iloc[:, 0], 
                         confidence_intervals.iloc[:, 1], 
                         color='k', alpha=0.1, label='Intervalo de Confianza')
        plt.scatter(pred_values.index, pred_values, color='r', marker='o', label='Predicción')
        plt.xlabel('Fecha')
        plt.ylabel('Valores')
        plt.title('Predicción de 3 meses y errores estimados')
        plt.legend(loc='upper left')
        plt.grid(True)
        plt.ylim(0)
        plt.tight_layout()

        filepath = f'Images_output/pred_{model_id}.png'
        plt.savefig(filepath)  # Guardar el gráfico como imagen PNG
        return filepath
    except Exception as e:
        print(f"Error making predictions: {e}")

# Función para generar un número hash a partir de una palabra
def generate_number_from_word(word):
    # Crear un objeto hash SHA-256
    hash_object = hashlib.sha256(word.encode())
    
    # Obtener el hash hexadecimal
    hex_hash = hash_object.hexdigest()
    
    # Convertir el hash hexadecimal a un número entero
    num_hash = int(hex_hash, 16)
    
    # Obtener un número de 10 cifras como máximo
    num_10_digits = num_hash % (10**10)
    
    return num_10_digits

# Código principal para la ejecución directa del script
if __name__ == '__main__':
    data_file_path = r'Datawarehouse\pizzas_normales_sd.csv'
    future_steps = 3

    # Cargar datos
    historical_data = load_data(data_file_path)

    # Preparar los datos para el modelo SARIMA
    y = historical_data

    # Construir y entrenar el modelo
    model_fit = build_model(y)

    # Guardar el modelo entrenado
    model_id = save_model(model_fit)

    # Cargar el modelo guardado
    loaded_model = load_model(model_id)

    # Hacer predicciones futuras
    future_predictions, confidence_intervals = make_predictions(loaded_model, future_steps)
    print(f'Predicciones futuras: {future_predictions}')
    print(f'Intervalos de confianza: {confidence_intervals}')
