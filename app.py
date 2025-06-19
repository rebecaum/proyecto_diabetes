import streamlit as st
import joblib
import pandas as pd

# Título principal
st.title("🩺 Predicción de Diabetes - Evaluación Clínica")

st.markdown(
    "Introduce los datos clínicos del paciente para estimar el riesgo de diabetes "
    "usando un modelo validado basado en Machine Learning."
)

# Formulario lateral
st.sidebar.header("📋 Datos del paciente")

insulina = st.sidebar.number_input(
    label="Insulina (uU/mL)",
    min_value=0.0, max_value=900.0, value=30.5,
    help="Nivel de insulina en sangre. Valores altos pueden indicar resistencia a la insulina."
)

pedigri = st.sidebar.number_input(
    label="Función Pedigrí de Diabetes (DPF)",
    min_value=0.0, max_value=2.5, value=0.37,
    help="Índice de predisposición genética. Cuanto mayor, mayor riesgo."
)

edad = st.sidebar.number_input(
    label="Edad",
    min_value=21, max_value=100, value=29,
    help="Edad del paciente. El riesgo aumenta con la edad."
)

embarazos = st.sidebar.number_input(
    label="Número de embarazos",
    min_value=0, max_value=20, value=3,
    help="Número total de embarazos. Relevante en mujeres para evaluación de riesgo."
)

glucosa = st.sidebar.number_input(
    label="Glucosa (mg/dL)",
    min_value=0, max_value=250, value=117,
    help="Nivel de glucosa en ayunas. Valores altos indican riesgo de diabetes."
)

presion = st.sidebar.number_input(
    label="Presión arterial (mm Hg)",
    min_value=0, max_value=160, value=72,
    help="Presión arterial diastólica. Importante en control metabólico."
)

piel = st.sidebar.number_input(
    label="Grosor del pliegue cutáneo (mm)",
    min_value=0, max_value=100, value=23,
    help="Medida del tejido subcutáneo. Asociado a masa grasa corporal."
)

imc = st.sidebar.number_input(
    label="IMC (Índice de Masa Corporal)",
    min_value=10.0, max_value=70.0, value=32.0,
    help="Relación peso/talla. IMC > 30 indica obesidad."
)

# Cargar pipeline y modelo
@st.cache_resource
def cargar_modelos():
    pipeline = joblib.load("pipelines/pipeline_v1.pkl")
    modelo = joblib.load("models/svc_opt_smote_v1_umbral035.pkl")
    return pipeline, modelo

pipeline, modelo = cargar_modelos()

# Botón de predicción
if st.button("🔍 Evaluar riesgo"):
    datos = pd.DataFrame([{
        'Insulin': insulina,
        'DiabetesPedigreeFunction': pedigri,
        'Age': edad,
        'Pregnancies': embarazos,
        'Glucose': glucosa,
        'BloodPressure': presion,
        'SkinThickness': piel,
        'BMI': imc
    }])

    datos_transformados = pipeline.transform(datos)
    pred = modelo.predict(datos_transformados)[0]
    proba = modelo.predict_proba(datos_transformados)[0][1]

    st.subheader("🧾 Resultado del modelo:")
    if pred == 1:
        st.error(f"⚠️ Riesgo de diabetes detectado ({proba:.2%})")
    else:
        st.success(f"✅ No se detecta riesgo aparente ({proba:.2%})")

