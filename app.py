import streamlit as st
import joblib
import pandas as pd
import datetime
import base64
from fpdf import FPDF
import os

# === CONFIGURACIÓN INICIAL ===
st.set_page_config(page_title="Predicción de Diabetes")
correo_lider = "lider@dominio.com"
correos_profesionales_autorizados = [
    "ana@centro1.com",
    "david@centro2.org",
    "maria@clinica3.es"
]
CSV_PATH = "resultados/datos_pacientes.csv"
os.makedirs("resultados", exist_ok=True)

# === CLASE PARA MODELO CON UMBRAL ===
class ModeloConUmbral:
    def __init__(self, modelo, umbral=0.5):
        self.modelo = modelo
        self.umbral = umbral

    def predict(self, X):
        return (self.modelo.predict_proba(X)[:, 1] >= self.umbral).astype(int)

    def predict_proba(self, X):
        return self.modelo.predict_proba(X)

# === CARGA DEL MODELO ===
@st.cache_resource
def cargar_modelos():
    pipeline = joblib.load("pipelines/pipeline_v1_fitted.pkl")
    modelo_base = joblib.load("model_pkl/svc_opt_smote_v1_umbral035.pkl")
    modelo = ModeloConUmbral(modelo_base, umbral=0.35)
    return pipeline, modelo

pipeline, modelo = cargar_modelos()

# === INTERFAZ PRINCIPAL ===
st.title("Aplicación de Predicción de Diabetes")
st.markdown("Introduce los datos clínicos del paciente para estimar el riesgo de diabetes y generar un informe.")
st.info("👉 Para comenzar, abre la barra lateral (icono ▸ arriba a la izquierda) y completa los datos del paciente.")

# === FORMULARIO CLÍNICO ===
st.sidebar.header("Datos del paciente")

nombre = st.sidebar.text_input("Nombre del paciente (opcional)")
profesional = st.sidebar.text_input("Profesional/Centro (opcional)")
email = st.sidebar.text_input("Correo electrónico del profesional (obligatorio para guardar)")

st.sidebar.write("---")
st.sidebar.subheader(" Correos de ejemplo para probar la app")
st.sidebar.write("**Líder:**")
st.sidebar.code("lider@dominio.com", language="none")
st.sidebar.write("**Profesionales autorizados:**")
st.sidebar.code("ana@centro1.com", language="none")
st.sidebar.code("david@centro2.org", language="none")
st.sidebar.code("maria@clinica3.es", language="none")
st.sidebar.info("Puedes copiar y pegar uno de estos correos en los campos de la app para simular los roles.")
st.sidebar.write("---")


insulina = st.sidebar.number_input("Insulina (uU/mL)", 0.0, 900.0, 30.5,
    help="Nivel de insulina en sangre. Valores altos pueden indicar resistencia a la insulina.")
pedigri = st.sidebar.number_input("Pedigrí de diabetes (DPF)", 0.0, 2.5, 0.37,
    help="Índice de predisposición genética. Cuanto mayor, mayor riesgo.")
edad = st.sidebar.number_input("Edad", 21, 100, 29,
    help="Edad del paciente. El riesgo aumenta con la edad.")
embarazos = st.sidebar.number_input("Nº de embarazos", 0, 20, 3,
    help="Solo mujeres. Puede estar asociado al riesgo de diabetes.")
glucosa = st.sidebar.number_input("Glucosa (mg/dL)", 0, 250, 117,
    help="Glucosa en ayunas. >125 puede indicar diabetes.")
presion = st.sidebar.number_input("Presión arterial (mm Hg)", 0, 160, 72,
    help="Presión diastólica. Relacionada con salud metabólica.")
piel = st.sidebar.number_input("Grosor del pliegue cutáneo (mm)", 0, 100, 23,
    help="Tejido subcutáneo. Relacionado con grasa corporal.")
imc = st.sidebar.number_input("IMC (kg/m²)", 10.0, 70.0, 32.0,
    help="Índice de Masa Corporal. >30 indica obesidad.")

st.markdown("Completa los datos y pulsa el botón para obtener el resultado.")

# === BOTÓN DE EVALUACIÓN ===
if st.button("Evaluar riesgo de diabetes"):
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

    st.subheader("Resultado del modelo:")
    if pred == 1:
        st.error(f"Riesgo de diabetes detectado ({proba:.2%})")
        resultado = "RIESGO DETECTADO"
    else:
        st.success(f"No se detecta riesgo aparente ({proba:.2%})")
        resultado = "SIN RIESGO"

    st.markdown("Datos introducidos:")
    st.dataframe(datos.T.rename(columns={0: "Valor"}))

    satisfaccion = st.radio("¿Cómo valoras la utilidad del resultado?",
                             ["Muy útil", "Útil", "Regular", "Poco útil", "Nada útil"])
    mejora = st.slider("¿Cuánto ha mejorado tu comprensión del riesgo?", 0, 10, 5)

    # === PDF ===
    fecha_actual = datetime.datetime.now().strftime("%d/%m/%Y")
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(200, 10, "Informe de Predicción de Diabetes", ln=True, align='C')

    pdf.set_font("Arial", '', 12)
    pdf.ln(10)
    pdf.cell(200, 10, f"Fecha de evaluación: {fecha_actual}", ln=True)
    if nombre:
        pdf.cell(200, 10, f"Nombre del paciente: {nombre}", ln=True)
    if profesional:
        pdf.cell(200, 10, f"Profesional/Centro: {profesional}", ln=True)
    if email:
        pdf.cell(200, 10, f"Correo electrónico: {email}", ln=True)

    pdf.ln(8)
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(200, 10, "Datos clínicos introducidos:", ln=True)
    pdf.set_font("Arial", '', 11)
    for col, val in datos.iloc[0].items():
        pdf.cell(200, 8, f"{col}: {val}", ln=True)

    pdf.ln(6)
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(200, 10, "Resultado del modelo:", ln=True)
    pdf.set_font("Arial", '', 11)
    if pred == 1:
        pdf.set_text_color(200, 0, 0)
        pdf.cell(200, 10, f"Riesgo de diabetes detectado: {proba:.2%}", ln=True)
    else:
        pdf.set_text_color(0, 128, 0)
        pdf.cell(200, 10, f"No se detecta riesgo aparente: {proba:.2%}", ln=True)
    pdf.set_text_color(0, 0, 0)

    pdf.ln(6)
    pdf.set_font("Arial", '', 11)
    pdf.cell(200, 10, f"Satisfacción: {satisfaccion}", ln=True)
    pdf.cell(200, 10, f"Nivel de comprensión: {mejora}/10", ln=True)

    temp_filename = "temp_informe.pdf"
    pdf.output(temp_filename)

    with open(temp_filename, "rb") as f:
        pdf_data = f.read()
    b64 = base64.b64encode(pdf_data).decode()
    os.remove(temp_filename)

    href = f'<a href="data:application/octet-stream;base64,{b64}" download="informe_diabetes.pdf">Descargar informe PDF</a>'
    st.markdown(href, unsafe_allow_html=True)

    # === GUARDAR EN CSV ===
    fila = datos.copy()
    fila["Fecha"] = fecha_actual
    fila["Resultado"] = resultado
    fila["Probabilidad"] = round(proba, 4)
    fila["Nombre"] = nombre
    fila["Profesional"] = profesional
    fila["Email"] = email
    fila["Satisfacción"] = satisfaccion
    fila["Mejora"] = mejora

    columnas_finales = ['Insulin', 'DiabetesPedigreeFunction', 'Age', 'Pregnancies',
                        'Glucose', 'BloodPressure', 'SkinThickness', 'BMI',
                        'Fecha', 'Resultado', 'Probabilidad',
                        'Nombre', 'Profesional', 'Email',
                        'Satisfacción', 'Mejora']

    if os.path.exists(CSV_PATH):
        df_existente = pd.read_csv(CSV_PATH)
        for col in fila.columns:
            if col not in df_existente.columns:
                df_existente[col] = ""
        df_final = pd.concat([df_existente, fila], ignore_index=True)
        for col in columnas_finales:
            if col not in df_final.columns:
                df_final[col] = ""
        df_final = df_final[columnas_finales]
    else:
        df_final = fila[columnas_finales]

    df_final.to_csv(CSV_PATH, index=False)

# === ACCESO CSV LÍDER ===
st.markdown("---")
st.header("Acceso al CSV acumulado (Líder)")

correo_introducido = st.text_input("Introduce el correo del líder para ver todos los datos")

if correo_introducido == correo_lider:
    st.success("Acceso concedido. Puedes descargar los datos acumulados.")
    if os.path.exists(CSV_PATH):
        df = pd.read_csv(CSV_PATH)
        csv_base64 = base64.b64encode(df.to_csv(index=False).encode()).decode()
        csv_link = f'<a href="data:file/csv;base64,{csv_base64}" download="datos_pacientes.csv">Descargar CSV acumulado</a>'
        st.markdown(csv_link, unsafe_allow_html=True)
elif correo_introducido:
    st.error("Correo no autorizado para acceso total.")

# === ACCESO CSV PROFESIONAL ===
st.markdown("---")
st.header("Acceso a tus registros (Profesional)")

correo_profesional = st.text_input("Introduce tu correo profesional para ver tus registros")

if correo_profesional:
    if correo_profesional in correos_profesionales_autorizados:
        if os.path.exists(CSV_PATH):
            df = pd.read_csv(CSV_PATH)
            df_filtrado = df[df["Email"] == correo_profesional]
            if not df_filtrado.empty:
                st.success(f"Se encontraron {len(df_filtrado)} registros asociados a {correo_profesional}.")
                st.dataframe(df_filtrado)

                csv_prof_base64 = base64.b64encode(df_filtrado.to_csv(index=False).encode()).decode()
                link_prof = f'<a href="data:file/csv;base64,{csv_prof_base64}" download="mis_registros.csv">Descargar mis registros</a>'
                st.markdown(link_prof, unsafe_allow_html=True)
            else:
                st.warning("No se encontraron registros asociados a este correo.")
    else:
        st.error("Este correo no está autorizado.")




