# Data Processing
import pandas as pd
import numpy as np

# Modelling
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.utils import resample
from joblib import dump
import joblib 

from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, ConfusionMatrixDisplay
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from scipy.stats import randint

df = pd.read_csv("training_dataset.csv")
df = df.drop(['Unnamed: 0', 'index','SK_ID_CURR'], axis=1)

# Prétraitement pour les variables numériques
numeric_features = [
    'CNT_CHILDREN', 'AMT_INCOME_TOTAL', 'AMT_CREDIT', 'AMT_ANNUITY', 'AMT_GOODS_PRICE',
    'CNT_FAM_MEMBERS', 'EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3', 'COUNT_DOCUMENT', 
    'PERSONAL_INVESTMENT', 'REPAYMENT_TERM', 'AGE', 'AGE_WORK', 'LABEL_PA', 'AMT_BALANCE', 
    'AMT_CREDIT_LIMIT_ACTUAL', 'AMT_DRAWINGS_ATM_CURRENT', 'AMT_PAYMENT_CURRENT', 'CREDIT_ACTIVE_counts',
    'DAYS_CREDIT_min', 'DAYS_CREDIT_ENDDATE_MAX', 'CNT_CREDIT_PROLONG_count', 'AMT_CREDIT_SUM_active', 
    'AMT_CREDIT_SUM_DEBT_active', 'AMT_CREDIT_SUM_OVERDUE_sum'
]
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

# Prétraitement pour les variables catégorielles
categorical_features = [
    'NAME_CONTRACT_TYPE', 'CODE_GENDER', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY', 
    'NAME_TYPE_SUITE', 'NAME_INCOME_TYPE', 'NAME_EDUCATION_TYPE', 'NAME_FAMILY_STATUS', 
    'NAME_HOUSING_TYPE', 'OCCUPATION_TYPE'
]
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='Missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

X = df.drop('TARGET', axis=1)
y = df['TARGET']

# Sous-échantillonnez la classe majoritaire
X_resampled, y_resampled = resample(X[y == 0], y[y == 0], n_samples=20000, random_state=42)

# Concaténez les données sous-échantillonnées avec la classe minoritaire
X_balanced = np.vstack((X_resampled, X[y == 1]))
y_balanced = np.hstack((y_resampled, y[y == 1]))

# Convertir les données concaténées en un DataFrame pandas
X_balanced_df = pd.DataFrame(X_balanced, columns=X.columns)

# Diviser les données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X_balanced_df, y_balanced, test_size=0.2)

# Créer le ColumnTransformer pour appliquer les pipelines de prétraitement
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ]
)

# Créer le pipeline global en chaînant le prétraitement avec un modèle d'apprentissage automatique
pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                           ('classifier', RandomForestClassifier(class_weight='balanced'))])

# Appliquer le pipeline à vos données d'entraînement
pipeline.fit(X_train, y_train)

# Faire des prédictions sur les données de test
y_pred = pipeline.predict(X_test)

cm = confusion_matrix(y_test, y_pred)

# Calculer l'accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Calculer la précision
precision = precision_score(y_test, y_pred)
print("Precision:", precision)

# Calculer le rappel (recall)
recall = recall_score(y_test, y_pred)
print("Recall:", recall)

# Afficher la matrice de confusion
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()

# Sauvegarde du modèle
dump(pipeline, 'credit_model.joblib')

# Charger le modèle
model = joblib.load('credit_model.joblib')


# Exemple d'utilisation
sk_id_curr = 100002  # Remplacez par l'ID réel du dataset