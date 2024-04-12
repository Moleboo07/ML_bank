import streamlit as st
import pandas as pd
import joblib 


# BACKEND 
model = joblib.load("credit_model.joblib")

data = pd.read_csv('training_dataset.csv')

def make_prediction_by_id(sk_id_curr):
    # Filtrer pour obtenir la ligne correspondante au SK_ID_CURR
    input_data = data[data['SK_ID_CURR'] == sk_id_curr]

    # Supprimer les colonnes non utilisées dans le modèle
    input_data = input_data.drop(['TARGET', 'SK_ID_CURR'], axis=1)

    # Vérifier si la ligne existe
    if input_data.empty:
        return "Aucune donnée trouvée pour cet ID"

    # Faire la prédiction
    prediction = model.predict(input_data)
    probability = model.predict_proba(input_data)[:, 1]  # Probabilité de la classe '1'

    probability_false = 1 -probability

    if prediction[0] == 1:
        return f"Prêt refusé avec une probabilité de {probability[0]:.2f}"
    else:
        return "Prêt accordé avec une probabilité de " + str(probability_false)


# FRONTEND 


st.title('ML BANK')

texte_utilisateur = st.text_input("Entrez l'id du client ici")
# texte_utilisateur = int(texte_utilisateur)
button_lancer_model = st.button('Lancer le scoring')


if button_lancer_model:
    texte_utilisateur = int(texte_utilisateur)
    st.write(make_prediction_by_id(sk_id_curr=texte_utilisateur))
