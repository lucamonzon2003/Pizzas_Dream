# Predicción de Unidades Vendidas

Esta aplicación predice las unidades vendidas para los próximos 3 meses utilizando modelos de series temporales implementados con Streamlit.

## Tabla de Contenidos

1. [Descripción del Proyecto](#descripción-del-proyecto)
2. [Instalación](#instalación)
3. [Uso](#uso)
4. [Documentación](#documentación)
5. [Contribuir](#contribuir)
6. [Licencia](#licencia)
7. [Contacto](#contacto)

## Descripción del Proyecto

Esta aplicación está diseñada para ayudar a predecir las unidades vendidas de productos, usando modelos de series temporales. La interfaz de usuario está construida con Streamlit, permitiendo a los usuarios cargar datos históricos, entrenar modelos, guardar y cargar modelos, y visualizar predicciones con sus respectivos intervalos de confianza.

## Instalación

### Requisitos Previos

- Python 3.7 o superior
- pip

### Instrucciones de Instalación

1. Clona este repositorio:
    ```bash
    git clone https://github.com/tu_usuario/tu_proyecto.git
    ```
2. Navega al directorio del proyecto:
    ```bash
    cd tu_proyecto
    ```
3. Instala las dependencias:
    ```bash
    pip install -r requirements.txt
    ```

## Uso

1. Ejecuta la aplicación:
    ```bash
    streamlit run app.py
    ```
2. Carga tus datos históricos en formato CSV y sigue las instrucciones en la interfaz.

### Ejemplo de Uso

1. Selecciona "Crear modelo" o "Cargar modelo" en la interfaz.
2. Si eliges "Crear modelo":
    - Asigna un identificador al modelo.
    - Sube el archivo CSV con los datos históricos.
    - Entrena el modelo.
    - Guarda el modelo.
3. Si eliges "Cargar modelo":
    - Introduce el hash del modelo guardado.
    - Sube el archivo CSV con los datos históricos.
    - Carga el modelo y genera predicciones.
4. Visualiza las predicciones y los intervalos de confianza en la interfaz.

## Documentación

Para más detalles sobre cómo funciona el proyecto, consulta la [documentación completa](docs/documentacion.md).

## Contribuir

¡Contribuciones son bienvenidas! Por favor, sigue los siguientes pasos:

1. Haz un fork del repositorio.
2. Crea una rama con tu nueva característica (`git checkout -b feature/nueva-caracteristica`).
3. Haz commit de tus cambios (`git commit -am 'Agrega nueva característica'`).
4. Haz push a la rama (`git push origin feature/nueva-caracteristica`).
5. Abre un Pull Request.

## Licencia

Distribuido bajo la licencia MIT. Ver `LICENSE` para más información.

## Contacto

Luca Hector Monzon - [tu_email@example.com](mailto:tu_email@example.com)

Enlace al Proyecto: https://github.com/tu_usuario/tu_proyecto
