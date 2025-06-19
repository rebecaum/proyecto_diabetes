import streamlit as st
import joblib
import pandas as pd
import datetime
import base64
from io import BytesIO
from fpdf import FPDF

# === Configuración inicial ===
st.set_page_config(page_title="Predicción de Diabetes", page_icon="🩺")

# === Cargar modelos ===
@st.cache_resource
def cargar_modelos():
    pipeline = joblib.load("pipelines/pipeline_v1.pkl")
    modelo = joblib.load("model_pkl/svc_opt_smote_v1_umbral035.pkl")
    return pipeline, modelo

pipeline, modelo = cargar_modelos()
fecha_actual = datetime.datetime.now().strftime("%d/%m/%Y")

# === Título y descripción ===
st.title("🩺 Aplicación de Predicción de Diabetes")
st.markdown("""
Esta herramienta permite estimar el **riesgo de diabetes** a partir de datos clínicos.
Introduce la información del paciente para obtener un informe descargable con los resultados.
""")

# === Datos de Identificación ===
with st.sidebar:
    st.header(":bust_in_silhouette: Identificación del paciente")
    nombre = st.text_input("Nombre del paciente (opcional)")
    profesional = st.text_input("Profesional/Centro (opcional)")
    st.markdown("*Puedes dejar los campos vacíos para una evaluación anónima.*")

# === Formulario clínico ===
st.sidebar.header("📋 Datos clínicos del paciente")

insulina = st.sidebar.number_input("Insulina (uU/mL)", 0.0, 900.0, 30.5,
    help="Nivel de insulina en sangre. Valores altos pueden indicar resistencia a la insulina.")
pedigri = st.sidebar.number_input("Pedigrí de diabetes (DPF)", 0.0, 2.5, 0.37,
    help="Índice de predisposición genética. Cuanto mayor, mayor riesgo.")
edad = st.sidebar.number_input("Edad", 21, 100, 29,
    help="Edad del paciente. El riesgo aumenta con la edad.")
embarazos = st.sidebar.number_input("Nº de embarazos", 0, 20, 3,
    help="Solo mujeres. Puede estar asociado al riesgo.")
glucosa = st.sidebar.number_input("Glucosa (mg/dL)", 0, 250, 117,
    help="Glucosa en ayunas. >125 puede indicar diabetes.")
presion = st.sidebar.number_input("Presión arterial (mm Hg)", 0, 160, 72,
    help="Presión diastólica. Relacionada con salud metabólica.")
piel = st.sidebar.number_input("Grosor del pliegue cutáneo (mm)", 0, 100, 23,
    help="Tejido subcutáneo. Relacionado con grasa corporal.")
imc = st.sidebar.number_input("IMC (kg/m²)", 10.0, 70.0, 32.0,
    help="Índice de Masa Corporal. >30 indica obesidad.")

# === Botón de predicción ===
if st.button("🔍 Evaluar riesgo de diabetes"):
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

    # === Resultado ===
    st.subheader("📊 Resultado del modelo:")
    if pred == 1:
        st.error(f"⚠️ Riesgo de diabetes detectado ({proba:.2%})")
        resultado = "RIESGO DETECTADO"
        color = (220, 20, 60)
        emoji = "🔴"
    else:
        st.success(f"✅ No se detecta riesgo aparente ({proba:.2%})")
        resultado = "SIN RIESGO"
        color = (0, 128, 0)
        emoji = "🟢"

    # === Mostrar datos ===
    st.markdown("### 📑 Datos introducidos")
    st.dataframe(datos.T.rename(columns={0: "Valor"}))

    # === Exportar CSV ===
    csv = datos.to_csv(index=False).encode('utf-8')
    st.download_button("📥 Descargar datos en CSV", data=csv, file_name="datos_diabetes.csv", mime='text/csv')

    # === Generar PDF ===
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", "B", 16)
    pdf.cell(200, 10, "Informe de Predicción de Diabetes", ln=True, align='C')
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, f"Fecha: {fecha_actual}", ln=True)
    if nombre:
        pdf.cell(200, 10, f"Paciente: {nombre}", ln=True)
    if profesional:
        pdf.cell(200, 10, f"Profesional/Centro: {profesional}", ln=True)
    pdf.ln(8)
    for col, val in datos.iloc[0].items():
        pdf.cell(200, 8, f"{col}: {val}", ln=True)
    pdf.ln(5)
    pdf.set_text_color(*color)
    pdf.cell(200, 10, f"Resultado: {resultado} ({proba:.2%})", ln=True)
    pdf.set_text_color(0, 0, 0)

    buffer = BytesIO()
    pdf.output(buffer)
    buffer.seek(0)
    b64 = base64.b64encode(buffer.read()).decode()
    link_pdf = f'<a href="data:application/octet-stream;base64,{b64}" download="informe_diabetes.pdf">📄 Descargar informe en PDF</a>'
    st.markdown(link_pdf, unsafe_allow_html=True)

    # === Valoración ===
    st.markdown("### 📝 Valoración del resultado:")
    satisfaccion = st.radio("¿Cómo valoras la utilidad del resultado?",
                             ["🔴 Nada útil", "🟠 Poco útil", "🟡 Regular", "🟢 Útil", "🟢 Excelente"])
    mejora = st.slider("¿Cuánto ha mejorado tu comprensión del riesgo?", 0, 10, 5)
    st.success(f"Gracias por tu valoración: {satisfaccion} | Nivel de mejora: {mejora}/10")

    # === Email (preparado) ===
    st.markdown("### ✉️ ¿Deseas recibir el informe por correo?")
    email = st.text_input("Introduce tu email (opcional):")
    if email:
        st.info(f"📧 El envío automático por correo estará disponible próximamente.")

    # === Cierre ===
    st.markdown("---")
    st.markdown("Gracias por usar esta herramienta. ¡Tu salud es lo más importante 💙!")
    st.markdown("Nos importa tu opinión 💬")

