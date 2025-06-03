import streamlit as st
import numpy as np
import pickle

# Cargar modelo
with open('tree_classifier_crit-entro_maxdepth-5_minleaf-4_minsplit2_42.sav', 'rb') as f:
    model = pickle.load(f)

# Título de la app
st.title("Predicción Diabetes")

# Inputs del usuario
n_embarazos = st.number_input('N° de embarazos', min_value=0, step=1)
nivel_glucosa = st.number_input('Nivel de glucosa en sangre', min_value=0.0, step=0.1, format="%.1f")
presion_sangre = st.number_input('Presion Sanginea', min_value=0.0, step=1.0, format="%.1f")
nivel_insulina = st.number_input('Insulina en sangre', min_value=0.0, step=1.0, format="%.1f")
bmi = st.number_input('Indice de masa Corporal', min_value=0.0, step=1.0, format="%.1f")
dpf = st.number_input('Pedegre de Diabetes', min_value=0.0, step=1.0, format="%.1f")
edad = st.slider('Edad', min_value=0, max_value=100, step=1)


# Botón para predecir
if st.button('Predecir'):
    # Preparar datos
    features = np.array([[int(n_embarazos), float(nivel_glucosa), float(presion_sangre), 
                      float(nivel_insulina), float(bmi), float(dpf), int(edad)]])
    prediction = model.predict(features)
    labels = ["Negativo", "Positivo"]
    st.success(f'El pronostico es: {labels[prediction[0]]}')