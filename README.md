# **Proyecto de Predicción de Diabetes Tipo 2 en Mujeres Pima**

Este proyecto de Machine Learning tiene como objetivo **predecir el riesgo de diabetes tipo 2** a partir de variables clínicas recogidas en mujeres de la comunidad indígena Pima. Se ha desarrollado un flujo de trabajo completo, desde la exploración y preprocesamiento de datos hasta la evaluación de múltiples modelos, su optimización y la implementación final en una app de **Streamlit**

---
## Enlaces del proyecto

- Cuaderno Proyecto de predicción de Diabetes: Cuaderno del proyecto (.ipynb): [Proyecto_Prediccion_de_Diabetes.ipyn](notebook/Proyecto_Prediccion_de_Diabetes.ipynb)

- Versión HTML del cuaderno: [Proyecto_Prediccion_de_Diabetes.html](notebook/Proyecto_Prediccion_de_Diabetes.html)

- Repositorio en GitHub_ https://github.com/rebecaum/proyecto_diabetes

- Aplicación desplegada en Streamlit: https://proyectodiabetes-f5gzcuzggnykhmrbp2vjja.streamlit.app/
 
> ⚠ Nota: Los correos de profesionales y líder deben ser configurados por el administrador. Los que figuran en la app son ejemplos para pruebas.

---
## Contexto del problema

La diabetes tipo 2 representa un importante problema de salud pública, especialmente en poblaciones con factores de riesgo como la comunidad Pima. El dataset utilizado contiene información clínica de mujeres adultas de esta comunidad, lo que permite entrenar modelos de clasificación binaria para predecir la presencia o ausencia de diabetes.

---
## Dataset utilizado

El dataset original contiene 8 variables clínicas predictoras y una variable objetivo (Outcome) que indica si la persona presenta diabetes tipo 2 (1) o no (0).

### 🔍 Variables predictoras:
1. **Pregnancies**: Número de embarazos previos. Indicador relevante en mujeres adultas, relacionado con factores metabólicos.
2. **Glucose**: Concentración de glucosa en plasma en ayunas (mg/dL). Una de las variables más significativas para detectar riesgo de diabetes.
3. **BloodPressure**: Presión arterial diastólica (mm Hg). Indicador cardiovascular relacionado indirectamente con el riesgo metabólico.
4. **SkinThickness**: Espesor del pliegue cutáneo del tríceps (mm). Estimador indirecto de grasa subcutánea.
5. **Insulin**: Niveles de insulina en sangre (mu U/ml). Utilizado como estimación del nivel de resistencia a la insulina.
6. **BMI**: Índice de Masa Corporal (peso en kg / altura en m²). Altamente correlacionado con riesgo de enfermedades metabólicas.
7. **DiabetesPedigreeFunction**: Estimación de riesgo heredado según antecedentes familiares. Valores más altos indican mayor probabilidad de predisposición genética.
8. **Age**: Edad del paciente.

### Variable objetivo:

- **Outcome**: Resultado diagnóstico binario (0: No diabetes, 1: Diabetes).

 > Nota: El tratamiento y análisis de estos datos se realizó siguiendo buenas prácticas de anonimización, limpieza, imputación y escalado, documentado en el notebook principal del proyecto.

---
## Flujo del Proyecto

1. Carga y exploración del dataset clínico.
2. Análisis exploratorio automatizado (EDA) y con generación de informes visuale y resumen estadístico 
   → ver `/resultados/informe_eda_diabetes_20250617_1020.html`.
3. Preprocesamiento avanzado: 
	- Revisión de valores nulos o anómalos (0 en variables clínicas).
	- Imputación estadística de valores incorrectos.
	- Codificación, escalado y limpieza.limpieza, imputación, codificación.
4. Análisis multivariante de correlaciones entre las variables predictoras y con el Outcome.
5. Tratamiento avanzado de datos: outliers, normalización y transformaciones.
6. Construcción de pipelines personalizados:
	- pipeline_v1: imputación + escalado + modelo
	- pipeline_v2: pipeline alternativo con tratamiento distinto de nulos
	y comparación de dos pipelines (`pipeline_v1`, `pipeline_v2`).  
   → Se seleccionó **pipeline_v1** como definitivo.
7. Selección de variables importantes mediante técnicas como: 
	- SelectKBest
	- RandomForestClassifier  
   → Se creó una versión reducida con las 5 variables más relevantes.
8. Entrenamiento de múltiples modelos con ambos pipelines y la versión rducida:
   - Logistic Regression, Decision Tree, Random Forest, Gradient Boosting, XGBoost, SVC, KNN, GaussianNB.
9. Optimización de modelos con GridSearchCV (LogReg, XGBoost, SVC).
10. Validación cruzada (5-fold) para selección preliminar: 
	- Mejores resultados para **LogisticRegression** en recall.
11. Evaluación de distintos umbrales de decisión personalizados para optimizar métricas clínicas basadas sobre todo en recall:
	- LogReg: 0.20
	- SVC: 0.35
12. Aplicación de SMOTE para balancear clases en modelos.
13. Comparación de modelos con SMOTE + ajuste de umbral:
    - Mejores resultados para **SVC con umbral 0.35**, mostró el mejor rendimiento clínico general.
14. Exploraciones avanzadas:
    a. Análisis por subgrupos de riesgo según IMC:
	- Segmentación por IMC (≥30 / <30) 
	- Entrenamiento en los modelos Logistic Regression, SVC, XGBoost
    	- Aplicación de SMOTE segmentado (Total, Parcial y Class Weight Interno)
    b. Optimización con Optuna en los mejores modelos más prometedores segmentados con SMOTE por IMC:
	  → LogisticRegression IMC alto SMOTE total con Optuna
 	  → XGBoost IMC bajo SMOTE parcial con Optuna
15. Comparativa final de tres modelos clínicos optimizados.
	  → LogisticRegression IMC alto SMOTE total con Optuna
 	  → XGBoost IMC bajo SMOTE parcial con Optuna
	  → SVC no segmentado SMOTE total y ajuste umbral 0.35
16. Selección del modelo final: **SVC como modelo principal**, por su rendimiento equilibrado, facilidad de uso sin segmentación y buena sensibilidad.
17. Guardado de modelos, pipeline, métricas y visualizaciones.
18. Desarrollo de app interactiva en Streamlit para predicción e informe PDF.

---
## Modelos entrenados y estrategias

Se evaluaron tres modelos clínicamente relevantes:

- **LogisticRegression**  
  - IMC ALTO  
  - SMOTE TOTAL  
  - Optimizado con Optuna

- **XGBoost**  
  - IMC BAJO  
  - SMOTE PARCIAL  
  - Optimizado con Optuna

- **SVC (Support Vector Classifier)** (modelo final seleccionado)  
  - Dataset completo, NO segmentado  
  - SMOTE aplicado a todo el conjunto  
  - Ajuste de umbral a 0.35  
  - ✔️ Rendimiento equilibrado y aplicabilidad general

---

## Modelo final seleccionado

- El modelo SVC, entrenado sobre el conjunto completo con SMOTE y ajuste de umbral a 0.35, fue seleccionado como el modelo clínico final por su rendimiento equilibrado, su estabilidad ante distintos escenarios y su facilidad de implementación en producción sin necesidad de segmentación adicional.  
	- svc_opt_smote_v1_umbral035.pkl

- Los modelos segmentados (LogisticRegression y XGBoost) se mantienen como estrategias complementarias para escenarios personalizados de riesgo. Se han conservado para pruebas futuras:
	- xgb_opt_optuna_smote_parcial_bajo.pkl
	- lr_opt_optuna_smote_total_alto.pkl

---
## Métricas utilizadas
- Accuracy
- Precision
- Recall (Sensibilidad)  → Métrica clínica clave
- F1-score
- Matriz de confusión

---
## Resultados finales de modelos clínicos optimizados
- Logisitc Regression (IMC Alto)
	- Accuracy= 0.6989, Precision: 0.6818, Recall= 0.6818, F1-score= 0.6818, ROC-AUC: 0.7639
- XGBoost (IMC Bajo)
	- - Accuracy= 0.8361, Precision: 0.5000, Recall= 0.7000, F1-score= 0.5833, ROC-AUC: 0.8431
- SVC (Modelo final) 
	- Accuracy: 0.6948
	- Precision: 0.5422
	- Recall: 0.8333
	- F1-score: 0.6569
	- ROC-AUC: 0.8081
	- Matriz de confusión: [[62  38] 
				[ 9  45]]

---
### Herramientas y librerías empleadas
- Python 3.10+
- Scikit-learn
- XGBoost
- Optuna
- Pandas, Numpy
- Matplotlib, Seaborn
- Streamlit
- Imbalanced-learn
- fpdf

---
## Estructura del Proyecto

```bash
proyecto-diabetes/
├── app.py                        ← Versión con límites clínicos ampliados
├── app_libre.py                  ← Versión sin límites (libre)
├── README.md
├── requirements.txt
├── LICENSE
├── .gitignore
│
├── notebooks/
│   └── Proyecto_Prediccion_Diabetes.ipynb  ← Notebook principal (versión final) 
│
├── data/
│   ├── raw/
│   │   └── diabetes_informe.csv
│   └── processed/  ← Datos limpios, escalados y segmentados
│       ├── limpieza/
│       ├── escalado/
│       ├── segmentado/
│       ├── final/
│       └── original/
│
├── model_pkl/   ← Modelos finales guardados
│   ├── svc_opt_smote_v1_umbral035.pkl       ← 🧠 Modelo elegido (final)
│   ├── xgb_opt_optuna_smote_parcial_bajo.pkl  ← Modelo avanzado (IMC bajo)
│   └── lr_opt_optuna_smote_total_alto.pkl     ← Modelo avanzado (IMC alto)
│
├── pipelines/
│   └── pipeline_v1.pkl  ← Pipeline definitivo
│
├── resultados/
│   ├── informe_eda_diabetes_20250617_1020.html
│   ├── matriz_confusion_*.png
│   ├── predicciones_*.csv
│   ├── comparativas_metricas.csv
│   └── visualizaciones_*.png
│
├── src/
│   ├── __init__.py
│   ├── data_preprocessing.py
│   └── model_utils.py
```

---
## Instrucciones de uso local
1. Clona el repositorio:
	git clone https://github.com/rebecaum/proyecto_diabetes.git
2. Instala las dependencias:
	pip install -r requirements.txt
3. Ejecuta la aplicación:
	streamlit run app.py

---
## Aplicaciones web desplegadas
Este proyecto cuenta con dos versiones funcionales de la app desarrollada en Streamlit:

1. App Principal (con límites clínicos)
- Enlace: App principal https://rebecaum-proyecto-diabetes-app.streamlit.app/
- Archivo: app.py
- Uso recomendado: profesionales de la salud, simulaciones clínicas con validación de rangos realistas.
- Características:
	- Requiere introducir valores clínicos dentro de rangos controlados.
	- Genera predicción binaria (riesgo/no riesgo).
	- Permite guardar resultados en CSV.
	- Genera informe PDF.
	- Permite accesos diferenciados para profesionales y líderes.

2. App Alternativa (versión libre sin restricciones)
- Archivo: app_libre.py
- Uso recomendado: pruebas con valores extremos, investigación, análisis de sensibilidad del modelo.
- Características:
	- Permite ingresar cualquier valor, sin validaciones mínimas o máximas.
	- Ideal para estudios de casos límite o experimentación.
	- Guarda resultados en un archivo separado: datos_pacientes_libre.csv.

### Nota importante sobre los roles

Esta aplicación permite simular dos tipos de usuarios:

- **Profesional de la salud**: puede introducir los datos de pacientes y consultar sus propios registros.
- **Líder clínico**: puede visualizar los datos agregados de todos los profesionales.

Para activar las funcionalidades asociadas a cada rol, es necesario introducir un correo electrónico válido.

**Correos de ejemplo disponibles para pruebas**:

- Profesional: `profesional@dominio.com` 
	ana@centro1.com
	david@centro2.org
	maria@clinica3.es
- Líder: `lider@dominio.com`

⚠️ Estos correos pueden modificarse directamente en el código (app.py o app_libre.py) para adaptarse a un entorno real.
---
### Uso de la aplicación web
Las aplicaciones desarrolladas en Streamlit permiten evaluar el riesgo de diabetes de forma interactiva:
	- Accede al enlace público (desplegado en Streamlit Cloud).
	- Abre la **barra lateral izquierda** haciendo clic en el icono `▸` si no está visible.
	- Introduce los datos clínicos del paciente (pueden ser anónimos).
	- Pulsa **Evaluar riesgo de diabetes**.
	- Obtendrás:
 	  - Resultado inmediato del modelo.
 	  - Informe PDF descargable.
  	  - Registro automático en CSV (si se introduce correo profesional).
	- El profesional puede consultar sus propios registros.
	- El líder, al introducir su correo, puede descargar el acumulado global.

> El sistema permite un uso clínico orientado a la toma de decisiones y la mejora de comprensión del riesgo de diabetes.

---
 Autora

Rebeca Urriolabeitia

Profesional de la Salud con la especialidad en geriatría| Experta en gestión de casos complejos y cronicidad|Experta en dirección de centros sociosanitarios | Experiencia en atención centrada en la persona y modelos organizativos innovadores.|
Actualmente en formación en Ciencia de Datos, con espedial interés en análisis predictivo y aplicación de la IA en salud y bienestar.




