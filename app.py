
import streamlit as st
import pandas as pd
import joblib
from datetime import date

# Configuración de la página
st.set_page_config(
    page_title="Predicción de Rendimiento de Acciones",
    page_icon="📈",
    layout="centered"
)

st.title("📈 Predicción del Rendimiento de Acciones")

st.write(
    "Esta aplicación utiliza un modelo de Regresión Lineal para estimar "
    "el rendimiento de una acción para el día siguiente."
)

# Cargar modelo y columnas
@st.cache_resource
def cargar_modelo():
    modelo = joblib.load("modelo_regresion_lineal.joblib")
    columnas = joblib.load("columnas_modelo.joblib")
    return modelo, columnas

modelo, columnas_modelo = cargar_modelo()

st.header("Datos de entrada")

# Datos generales
fecha = st.date_input("Fecha", value=date.today())

ticker = st.selectbox(
    "Acción (Ticker)",
    ["ACME", "BETA", "GAMMA"]
)

# Datos financieros
open_price = st.number_input(
    "Precio de apertura (Open)",
    min_value=0.0,
    value=100.0
)

high = st.number_input(
    "Precio máximo (High)",
    min_value=0.0,
    value=105.0
)

low = st.number_input(
    "Precio mínimo (Low)",
    min_value=0.0,
    value=95.0
)

close = st.number_input(
    "Precio de cierre (Close)",
    min_value=0.0,
    value=102.0
)

volume = st.number_input(
    "Volumen (Volume)",
    min_value=0.0,
    value=1000000.0
)

st.subheader("Indicadores técnicos")

ma7 = st.number_input(
    "Media móvil 7 días (MA_7)",
    min_value=0.0,
    value=100.0
)

ma21 = st.number_input(
    "Media móvil 21 días (MA_21)",
    min_value=0.0,
    value=100.0
)

rsi = st.number_input(
    "RSI",
    min_value=0.0,
    max_value=100.0,
    value=50.0
)

macd = st.number_input(
    "MACD",
    value=0.0
)

close_lag1 = st.number_input(
    "Cierre del día anterior (Close_lag1)",
    min_value=0.0,
    value=100.0
)

# Predicción
if st.button("Realizar predicción"):

    fecha_pd = pd.Timestamp(fecha)
    day_of_week = fecha_pd.weekday()

    if day_of_week > 4:
        st.warning(
            "La fecha seleccionada corresponde a un fin de semana. "
            "Seleccione un día hábil."
        )

    elif low > high:
        st.error(
            "El precio mínimo (Low) no puede ser mayor que el precio máximo (High)."
        )

    else:

        nuevo_registro = pd.DataFrame([{
            "Open": open_price,
            "High": high,
            "Low": low,
            "Volume": volume,
            "MA_7": ma7,
            "MA_21": ma21,
            "RSI": rsi,
            "MACD": macd,
            "DayOfWeek": day_of_week,
            "Close_lag1": close_lag1,
            "Close": close,

            # Variables obtenidas automáticamente de la fecha
            "Year": fecha_pd.year,
            "Month": fecha_pd.month,
            "Day": fecha_pd.day,

            # One-Hot Encoding del Ticker
            "Ticker_ACME": int(ticker == "ACME"),
            "Ticker_BETA": int(ticker == "BETA"),
            "Ticker_GAMMA": int(ticker == "GAMMA")
        }])

        # Garantizar exactamente las mismas columnas del entrenamiento
        nuevo_registro = nuevo_registro.reindex(
            columns=columnas_modelo,
            fill_value=0
        )

        # Predicción
        prediccion = modelo.predict(nuevo_registro)[0]

        st.subheader("Resultado")

        st.metric(
            "NextDayReturn estimado",
            f"{prediccion:.6f}"
        )

        st.metric(
            "Rendimiento estimado",
            f"{prediccion * 100:.4f}%"
        )

        if prediccion > 0:
            st.success(
                "📈 El modelo estima un rendimiento positivo para el día siguiente."
            )

        elif prediccion < 0:
            st.error(
                "📉 El modelo estima un rendimiento negativo para el día siguiente."
            )

        else:
            st.info(
                "El modelo estima un rendimiento cercano a cero."
            )

        st.caption(
            "Modelo desarrollado con fines académicos. "
            "La predicción no constituye una recomendación de inversión."
        )
