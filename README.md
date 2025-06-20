## 🔗 Aplicación desplegada en Streamlit:
https://proyectodiabetes-f5gzcuzggnykhmrbp2vjja.streamlit.app/

# 🩺 Proyecto de Predicción de Diabetes

Este proyecto de Machine Learning tiene como objetivo predecir el riesgo de diabetes tipo 2 a partir de variables clínicas. Se han desarrollado múltiples modelos aplicando distintas estrategias de mejora como balanceo de clases con SMOTE, ajuste de umbral de decisión, segmentación por riesgo clínico (IMC), y optimización de hiperparámetros con Optuna. La solución final incluye una app en Streamlit lista para ser desplegada.

---

## 🔄 Flujo del Proyecto

1. Carga y exploración del dataset clínico.
2. Análisis exploratorio automatizado (EDA) y validación visual → ver `/resultados/informe_eda_diabetes_20250617_1020.html`.
3. Preprocesamiento avanzado: limpieza, imputación, codificación.
4. Análisis multivariante de correlaciones.
5. Tratamiento avanzado de datos: outliers y transformaciones
6. Construcción y comparación de dos pipelines (`pipeline_v1`, `pipeline_v2`).  
   → Se seleccionó **pipeline_v1** como definitivo.
7. Análisis de importancia de variables.  
   → Se creó una versión reducida con las 5 variables más relevantes.
8. Entrenamiento de múltiples modelos con ambos pipelines y la versión rducida:
   - Logistic Regression, Decision Tree, Random Forest, Gradient Boosting, XGBoost, SVC, KNN, GaussianNB.
9. Optimización de modelos con GridSearchCV (LogReg, XGBoost, SVC).
10. Validación cruzada (5-fold): mejores resultados para **LogisticRegression** en recall.
11. Evaluación de umbrales de decisión personalizados (LogReg: 0.20, SVC: 0.35).
12. Aplicación de SMOTE para balancear las clases.
13. Comparación de modelos con SMOTE + ajuste de umbral:
    - **SVC con umbral 0.35** mostró el mejor rendimiento clínico general.
14. Exploraciones avanzadas:
    - Segmentación por IMC (≥30 / <30) en los modelos Logistic Regression, SVC, XGBoost
    - Aplicación de SMOTE segmentado (Total, Parcial y Class Weight Interno)
    - Optimización con Optuna en los mejores modelos segmentados con SMOTE:
	  → LogisticRegression IMC alto SMOTE total con Optuna
 	  → XGBoost IMC bajo SMOTE parcial con Optuna
15. Comparativa final de tres modelos clínicos optimizados.
	  → LogisticRegression IMC alto SMOTE total con Optuna
 	  → XGBoost IMC bajo SMOTE parcial con Optuna
	  → SVC no segmentado SMOTE total y ajuste umbral 0.35
16. Elección de **SVC como modelo principal**, por su rendimiento estable, facilidad de uso sin segmentación y buena sensibilidad.
17. Guardado de modelos, pipeline, métricas y visualizaciones.
18. Desarrollo de app interactiva con Streamlit.

---

## 🧠 Modelos entrenados y estrategias

Se evaluaron tres modelos clínicamente relevantes:

- **LogisticRegression**  
  - IMC ALTO  
  - SMOTE TOTAL  
  - Optimizado con Optuna

- **XGBoost**  
  - IMC BAJO  
  - SMOTE PARCIAL  
  - Optimizado con Optuna

- **SVC (Support Vector Classifier** (modelo final seleccionado)  
  - Dataset completo, NO segmentado  
  - SMOTE aplicado a todo el conjunto  
  - Ajuste de umbral a 0.35  
  - ✔️ Rendimiento equilibrado y aplicabilidad general

---

## ✅ Modelo final seleccionado

- El modelo SVC, entrenado sobre el conjunto completo con SMOTE y ajuste de umbral a 0.35, fue seleccionado como el modelo clínico final por su rendimiento equilibrado, su estabilidad ante distintos escenarios y su facilidad de implementación en producción sin necesidad de segmentación adicional.  
	- svc_opt_smote_v1_umbral035.pkl

- Los modelos segmentados (LogisticRegression y XGBoost) se mantienen como estrategias complementarias para escenarios personalizados de riesgo. Se han conservado para pruebas futuras:
	- xgb_opt_optuna_smote_parcial_bajo.pkl
	- lr_opt_optuna_smote_total_alto.pkl

---

## 📁 Estructura del Proyecto

```bash
proyecto-diabetes/
├── app.py
├── README.md
├── requirements.txt
├── LICENSE
├── .gitignore
│
├── notebooks/
│   └── Proyecto_Prediccion_Diabetes.ipynb
│
├── data/
│   ├── raw/
│   │   └── diabetes_informe.csv
│   └── processed/
│       ├── limpieza/
│       ├── escalado/
│       ├── segmentado/
│       ├── final/
│       └── original/
│
├── model_pkl/
│   ├── svc_opt_smote_v1_umbral035.pkl       ← 🧠 Modelo elegido (final)
│   ├── xgb_opt_optuna_smote_parcial_bajo.pkl  ← Modelo avanzado (IMC bajo)
│   └── lr_opt_optuna_smote_total_alto.pkl     ← Modelo avanzado (IMC alto)
│
├── pipelines/
│   └── pipeline_v1.pkl
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

### 🚀 Uso de la aplicación web

La aplicación desarrollada en Streamlit permite evaluar el riesgo de diabetes de forma interactiva:

1. Accede al enlace público (desplegado en Streamlit Cloud).
2. Abre la **barra lateral izquierda** haciendo clic en el icono `▸` si no está visible.
3. Introduce los datos clínicos del paciente (pueden ser anónimos).
4. Pulsa **Evaluar riesgo de diabetes**.
5. Obtendrás:
   - Resultado inmediato del modelo.
   - Informe PDF descargable.
   - Registro automático en CSV (si se introduce correo profesional).
6. El profesional puede consultar sus propios registros.
7. El líder, al introducir su correo, puede descargar el acumulado global.

> El sistema permite un uso clínico orientado a la toma de decisiones y la mejora de comprensión del riesgo de diabetes.



