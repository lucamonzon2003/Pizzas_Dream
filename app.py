import streamlit as st
import pandas as pd
from PIL import Image
import time
import Pipeline as model

# Interfaz de Streamlit
st.title('Predicción de Unidades Vendidas')
st.write('Esta aplicación predice las unidades vendidas para los próximos 3 meses.')

# Cargar datos históricos de ejemplo
df_prueba = pd.read_csv(r'Datawarehouse\pizzas_normales_sd.csv')

# Selección de opción inicial: Crear modelo o Cargar modelo
options = ['Crear modelo', 'Cargar modelo']
init_option = st.selectbox('Selecciona una opción:', options, index=None)

if init_option == 'Crear modelo':
    # Entrada de texto para asignar un ID al modelo
    model_id = st.text_input('Asigna un identificador al modelo:', max_chars=10)

    if model_id:
        st.write('El archivo CSV debe tener el siguiente formato:')
        st.dataframe(df_prueba, use_container_width=True)

        # Cargar datos históricos desde archivo CSV
        data_file_path = st.file_uploader("Sube el archivo CSV con los datos históricos", type=["csv"])
        historical_data = model.load_data(data_file_path)

        if historical_data is not None:
            st.write('Datos históricos:')
            st.dataframe(historical_data, use_container_width=True)
            # Botón para entrenar el modelo
            if st.button('Entrenar modelo'):
                model_fit, avg_mse, mse = model.build_model(historical_data)
                if model_fit:
                    model_hash = model.save_model(model_fit, model_id)
                    if model_hash:
                        st.success(f'Modelo entrenado y guardado con éxito con el Hash: {model_hash}')
                        st.write(f'avg_mse del modelo: {avg_mse}')
                        st.write(f'mse del modelo: {mse}')
                    else:
                        st.error('Error al guardar el modelo.')
                else:
                    st.error('Error al entrenar el modelo.')

            # Entrada de texto para asignar un ID al modelo
            model_hash = st.text_input('Escribe el Hash:', max_chars=10)

            # Botón para cargar el modelo y hacer predicciones
            if st.button('Cargar modelo y hacer predicciones'):
                # Intentar cargar el modelo y hacer predicciones
                loaded_model = model.load_model(model_hash)
                if loaded_model:
                    st.write(f'ID del modelo: {model_hash}')
                    future_predictions, confidence_intervals = model.make_predictions(loaded_model, 3)

                    if future_predictions is not None and confidence_intervals is not None:
                        # Dividir la pantalla en dos columnas
                        col1, col2 = st.columns(2)

                        # Mostrar predicciones y intervalos de confianza en columnas
                        with col1:
                            st.write('Predicciones futuras:')
                            st.dataframe(future_predictions)

                        with col2:
                            st.write('Intervalos de confianza:')
                            st.dataframe(confidence_intervals)

                        # Visualizar el gráfico de predicciones
                        pred_path = model.visualize_predictions(historical_data, future_predictions, confidence_intervals, model_hash)
                        st.image(Image.open(pred_path), caption='Gráfico de Predicciones', use_column_width=True)
                    else:
                        st.error('Error al hacer las predicciones.')
                else:
                    st.error('Error al cargar el modelo.')
        else:
            st.warning('Por favor, sube un archivo CSV con los datos históricos.')
    else:
        timestamp = str(int(time.time()))
        timestamp = timestamp[-6:]
        st.info(f'Introduce un ID para el modelo o usa el que se genera automáticamente: {timestamp}')

elif init_option == 'Cargar modelo':
    # Entrada de texto para cargar el modelo por ID
    model_hash = st.text_input('Escribe el Hash del modelo:', max_chars=10)
    
    if model_hash:
        # Cargar datos históricos desde archivo CSV
        data_file_path = st.file_uploader("Sube el archivo CSV con los datos históricos", type=["csv"])
        historical_data = model.load_data(data_file_path)

    if model_hash:
        # Botón para cargar el modelo y hacer predicciones
        if st.button('Cargar modelo y hacer predicciones'):
            # Intentar cargar el modelo y hacer predicciones
            loaded_model = model.load_model(model_hash)
            if loaded_model:
                st.write(f'Hash del modelo: {model_hash}')
                future_predictions, confidence_intervals = model.make_predictions(loaded_model, 3)

                if future_predictions is not None and confidence_intervals is not None:
                    # Dividir la pantalla en dos columnas
                    col1, col2 = st.columns(2)

                    # Mostrar predicciones y intervalos de confianza en columnas
                    with col1:
                        st.write('Predicciones futuras:')
                        st.dataframe(future_predictions)

                    with col2:
                        st.write('Intervalos de confianza:')
                        st.dataframe(confidence_intervals)

                    # Visualizar el gráfico de predicciones
                    pred_path = model.visualize_predictions(historical_data, future_predictions, confidence_intervals, model_hash)
                    st.image(Image.open(pred_path), caption='Gráfico de Predicciones', use_column_width=True)
                else:
                    st.error('Error al hacer las predicciones.')
            else:
                st.error('Error al cargar el modelo.')

else:
    st.warning('Selecciona una opción para comenzar.')

