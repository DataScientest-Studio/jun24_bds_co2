import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.tree import DecisionTreeRegressor
from joblib import dump

df = pd.read_csv(f'data_2012-2015.csv', on_bad_lines="skip", sep= ',', low_memory=False)

### PREPROCESSING ###

scaler = StandardScaler()
encoder_le =  LabelEncoder()

file = '/Users/louiserigal/Downloads/data_2012-2015.csv'
df_original = pd.read_csv(file, on_bad_lines="skip", sep= ',', low_memory=False)

liste_cbr = {"GO":"Gazole",
            "ES":"Essence",
            "EH":"Essence",
            "GH":"Gazole",
            "ES/GN":"Essence",
            "GN/ES":"Gaz Naturel Vehicule (GNV)",
            "ES/GP":"Essence",
            "GP/ES":"Gaz de Petrole Liquefie (GPL)",
            "EL":"Electrique",
            "GN":"Gaz Naturel Vehicule (GNV)",
            "EE":"Essence",
            "FE":"SuperEthanol-E85",
            "GL":"Gazole"}

df_original["Carburant"] = df_original["Carburant"].replace(liste_cbr)

df = df_original[["Consommation mixte (l/100km)", "Carburant", "CO2 (g/km)", "Puissance administrative","masse vide euro min (kg)"]]
df = df.dropna(how="any")

df["Carburant"] = encoder_le.fit_transform(df["Carburant"])

X = df.drop(columns="CO2 (g/km)")
X = scaler.fit_transform(X)
y = df["CO2 (g/km)"]

# TRAIN TEST SPLIT - 20% en test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=9001)

# Modélisation
model = DecisionTreeRegressor(max_depth = None)
model.fit(X_train, y_train)
dump(model, "decision_tree")