import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler, LabelEncoder
from keras import layers, models, optimizers

from joblib import load, dump
import streamlit as st

#################################
# Création de la page Streamlit #
#################################

st.title("Emission de CO2 par les véhicules")
st.sidebar.title("Sommaire")
pages=["Présentation du projet", "Exploration", "Data Visualisation", "Preprocessing",
    "Modélisations", "Interprétabilité", "Conclusions"]
page=st.sidebar.radio("Aller vers", pages)

### Page de présentation
if page == pages[0] : 
    st.header("Présentation du projet")
    st.write("Le transport routier contribue à environ un cinquième des émissions totales de l'Union européenne (UE) de dioxyde de carbone (CO2), le principal gaz à effet de serre (GES), dont 75 % proviennent des voitures particulières.")
    st.write('Notre objectif est d’identifier les différents facteurs et caractéristiques techniques jouant un rôle dans la pollution émise par les véhicules. Prédire à l’avance la pollution de certains types de véhicules est une information cruciale pour opérer une décarbonation de l’industrie automobile.')
    st.write("Nous disposons de données mises à disposition par l'European Environment Agency, pour les véhicules enregistrés au sein de l'UE, et par l'Agence de l'environnement et de la maîtrise de l'énergie, pour les véhicules français.")
    st.write('La volumétrie ainsi que l’absence notable de la variable associée à la consommation de carburant dans le jeu de données européen, nous conduit à privilégier la source données de l’ADEME. Le jeu de données retenu est constitué par les données disponibles en France entre 2012 et 2015, représentant 160 826 observations.')
    
### Exploration des données
file = '/Users/louiserigal/Downloads/data_2012-2015.csv'
df = pd.read_csv(file, on_bad_lines="skip", sep= ',', low_memory=False)
if page == pages[1] :
    st.header("Exploration des données")
    st.write('Nous nous intéressons aux données des véhicules enregistrés France entre 2012 et 2015.')
    st.subheader("Aperçu du jeu de données")
    st.dataframe(df.head(5))
    st.write("Il y a", df.shape[0], "observations dans notre dataset et", df.shape[1], "colonnes les caractérisant.")
    st.subheader("Informations principales sur le jeu de données")
    st.dataframe(df.describe())
    st.subheader("Valeurs manquantes")
    st.write("On observe assez peu de valeurs manquantes dans le dataset.")
    st.write("Deux variables en particulier présentent un grand nombre de valeurs manquantes (16 : HC (g/km) et 23 : Date de mise à jour). Ces variables seront donc supprimées. La présence d’un quantité non négligeable de valeurs manquantes dans les variables Carrosserie et gamme est provoquée par l’inclusion du jeu de données de 2015 (ces variables y sont absentes).")
    if st.checkbox("Afficher les NA") :
        st.dataframe(df.isna().sum())

### Visualisation
if page == pages[2] : 
    st.header("Data Vizualization")

    # Heatmap
    st.subheader('Heatmap')
    st.write("Afin de pouvoir déterminer plus facilement les variables numériques à cibler, il est possible de créer une heatmap. Un intérêt particulier sera donné aux variables ayant un fort degré de corrélation (le plus éloigné de 0) avec la variable cible : CO2 (g/km).")
    variables_num = df.select_dtypes(include = ['int64', 'float64'])
    cor = variables_num.corr()
    fig_heatmap = px.imshow(cor)
    st.plotly_chart(fig_heatmap) 
    st.write("Plusieurs variables sont corrélées avec la variable cible, notamment une, avec un degré de corrélation très élevé (0.97) : la Consommation mixte (l/100km), qui, comme son nom l’indique, donne la consommation en carburant du véhicule en litre pour 100 km (urbaine et extra-urbaine).") 
    st.write("Observons plus en détail la relation entre consommation mixte et émissions de CO2.")

    # Nuage de points conso mixte et CO2
    dictionnaire_carburant = {"GO":"Gazole",
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
    st.subheader('Nuage de points - émissions de CO2 (g/km) en fonction de la consommation mixte (l/100km) selon le carburant utilisé')
    df['Carburant'] = df['Carburant'].replace(dictionnaire_carburant)
    fig_scatter = px.scatter(df, x="Consommation mixte (l/100km)", y="CO2 (g/km)", color = 'Carburant',
                 title='CO2 émis selon la consommation de carburant mixte et le type de carburant utilisé')
    st.plotly_chart(fig_scatter) 
    st.write("Comme attendu, les points se regroupent de façon linéaire, ce qui signifie que cette variable nous sera utile pour prédire les émissions.")
    st.write("Toutefois, plusieurs droites semblent se dessiner. Cela indique donc qu'une variable supplémentaire affecte les résultats, certainement une variable catégorielle : le carburant.")
    st.write("La faible présence sur le graphique de véhicules utilisant un carburant autre que l’essence ou le gazole indique un potentiel déséquilibre dans le jeu de données.")
    st.write("Observons cela de plus près.")

    # Répartition des carburants
    st.subheader("Proportion de chaque type de motorisation")
    # Pie Chart
    occurence = []
    l = df['Carburant'].unique()
    for i in range (len(l)) :
        occurence.append(df.loc[df['Carburant'] == l[i]].count()[0])
    from plotly.subplots import make_subplots
    fig = make_subplots(rows = 1,
                    cols = 2,
                    specs=[[{'type':'domain'}, {'type':'domain'}]],
                    subplot_titles = ['Répartition globale', 'Zoom'],
                    )

    colors = ['lightblue','green','lightseagreen','antiquewhite','cadetblue','darkorange','goldenrod']
    fig.add_trace(go.Pie(labels = l,
                     values = occurence,
                     marker_line = dict(color = 'black', width = 1.5), # Couleur et épaisseur de la ligne
                     marker_colors = colors,  # Couleur de chaque partie du camembert
                     pull = [0,0.1,0,0,0,0],
                      name = 'Global'),
                      row = 1, col = 1)

    l_1 = np.delete(l, [0,1])
    occurence_sans_maj = []
    for i in range (len(l_1)) :
        occurence_sans_maj.append(df.loc[df['Carburant'] == l_1[i]].count()[0])

    fig.add_trace(go.Pie(labels = l_1,
                      values = occurence_sans_maj,
                    name = 'Zoom'),
                    row = 1, col = 2)

    fig.update_layout(title="Types de carburants des véhicules enregistrés en France entre 2012 et 2015",showlegend=True, legend_title = 'Légende')
    st.plotly_chart(fig) 
    st.write("Il y a une très forte représentation de véhicules utilisant du gazole comme carburant (84,2%), et dans une bien moindre mesure, les véhicules essence (15.3%). Les quatre autres motorisations ne représentent au final qu'un total de 0.5% du jeu de données restant. Ce déséquilibre entre les types de carburants présents peut entraîner un biais qui peut fausser les prédictions d’émission des véhicules utilisant ces types de carburants sous-représentés.")

    # Boîte à moustache
    st.subheader("Boîte à moustaches (Box plot) de l'émission de CO2 (g/km) en fonction du type de carburant")
    st.write("Le graphique ci-dessous doit nous permettre de vérifier la distribution des valeurs d’émissions des véhicules selon le type de carburant utilisé, afin, entre autre, de faire apparaître d’éventuelles valeurs aberrantes.")
    fig_boxplot = px.box(df, x = 'Carburant', y = 'CO2 (g/km)')
    st.plotly_chart(fig_boxplot)
    st.write("La présence de points hors des boîtes (notamment dans la catégorie gazole) indique la présence de valeurs éloignées du reste des autres valeurs. Toutefois, leurs écarts ne semblent pas significatifs, ce qui signifie que ces valeurs, bien que extrêmes, restent valables et peuvent donc être gardées dans le jeu de données.")

### Préprocessing
if page == pages[3] :
    st.header("Préprocessing")

    # Méthodo
    st.subheader("Méthodologie")
    st.write("Tout au long du projet, nous avons suivi une ligne directrice claire de simplicité qui nous a guidé dans les choix effectués.")
    st.write("La méthodologie suivie s’explique par le caractère extrêmement concret de la problématique d’émission de CO2 par un véhicule motorisé. La plupart des données de notre base font références à des concepts familiers pour tout le monde, et plus important encore, ces concepts accessibles semblent expliquer notre variable cible. Il nous a semblé important de conserver cette simplicité et lisibilité pour permettre d’obtenir un modèle explicatif et permettant d’agir en amont et en aval de prises de décision pour les acteurs potentiels (constructeur, utilisateur, administration réglementaire ou lobbying écologique…). Cette méthodologie est essentiellement constituée des deux principes suivants :")  
    st.write(" - Limiter si possible le nombre de variables du jeu de données sans compromettre la qualité des résultats de modélisation")
    st.write(" - Ne conserver si possible que des variables largement accessibles et compréhensibles (pour un utilisateur qui n’aurait pas accès à l’ensemble des données moteur, par exemple)")

    # Sélection des données
    st.subheader("Sélection des données")
    st.write('La sélection finale de données a porté sur les 4 variables suivantes : Puissance administrative, Consommation mixte (l/100km), Masse vide euro min (kg), Carburant.')
    st.write("Au cours de la modélisation, nous avons simplifié le nombre de variables processées au fur et à mesure que nous avons constaté que les résultats obtenus sur un nombre de variables restreint restaient équivalents à ceux sur un nombre de variables plus larges.")

    # Nettoyage et préparation des données
    st.subheader('Nettoyage et préparation des données')
    st.write("Le nettoyage des données est effectué en supprimant simplement les valeurs manquantes. La table passe ainsi de 160 826 entrées à 160 667, ce qui ne représente que 0.1% de données manquantes !")
    st.write("Notre seule variable catégorielle ‘Carburant’ comporte 13 occurences. Mais l’analyse précise de ces occurences permet de les regrouper en 5 catégories: Essence et Gazole, qui représentent la majorité des entrées, Gaz Naturel, Gaz de Pétrole Liquéfié (GPL) et Super-Ethanol-E85. Les données ont été renommées en conséquence, puis la variable a été transformée en variables d’état (OneHotEncoder fait simplement avec l’instruction pd.get_dummies). Et pour simplifier l’analyse des modèles, nous avons modifier les catégories de carburant en liste numérique (avec LabelEncoder).")
    st.write("Ensuite, la variable cible CO2 a été conservée comme telle, tandis que les 3 autres variables numériques ont été normalisées par l’application de la fonction StandardScaler, compte tenu des différentes échelles de ces métriques (entre 800 et 3000 pour le poids et entre 2 et 80 pour la puissance administrative aux extrêmes).")
    st.write("Enfin, les données ont été séparées en un échantillon d’entraînement et un échantillon de test de 20% - en utilisant le paramétrage de la génération de variables aléatoires afin de pouvoir comparer les différents modèles étudiées sur des bases comparables.")

################
# Modélisation #
################

class CustomRegression():
    def __init__(self):
            
        self.w_c1 = tf.Variable(tf.random.normal([1]), name='weight_carb_1')
        self.w_c2 = tf.Variable(tf.random.normal([1]), name='weight_carb_2')
        self.w_c3 = tf.Variable(tf.random.normal([1]), name='weight_carb_3')
        self.w_c4 = tf.Variable(tf.random.normal([1]), name='weight_carb_4')
        self.w_c5 = tf.Variable(tf.random.normal([1]), name='weight_carb_5')

        self.b_c1 = tf.Variable(tf.random.normal([1]), name='bias_carb_1')
        self.b_c2 = tf.Variable(tf.random.normal([1]), name='bias_carb_2')
        self.b_c3 = tf.Variable(tf.random.normal([1]), name='bias_carb_3')
        self.b_c4 = tf.Variable(tf.random.normal([1]), name='bias_carb_4')
        self.b_c5 = tf.Variable(tf.random.normal([1]), name='bias_carb_5')

    def __call__(self, conso, carb1, carb2, carb3, carb4, carb5):

        conso = tf.convert_to_tensor(conso, dtype=tf.float32)

        carb1 = tf.convert_to_tensor(carb1, dtype=tf.float32)
        carb2 = tf.convert_to_tensor(carb2, dtype=tf.float32)
        carb3 = tf.convert_to_tensor(carb3, dtype=tf.float32)
        carb4 = tf.convert_to_tensor(carb4, dtype=tf.float32)
        carb5 = tf.convert_to_tensor(carb5, dtype=tf.float32)
            
        return carb1 * (conso * self.w_c1 + self.b_c1) + carb2 * (conso * self.w_c2 + self.b_c2) + carb3 * (conso * self.w_c3 + self.b_c3) + carb4 * (conso * self.w_c4 + self.b_c4) + carb5 * (conso * self.w_c5 + self.b_c5)

def affichage_metrics(residus, y_pred, y_test):
    st.write("MSQE : {:.2f}".format(mean_squared_error(y_test, y_pred)))
    st.write("MAE : {:.2f}".format(mean_absolute_error(y_test, y_pred)))

    st.write("\nProportion < 1% d'ecart  : {:.2f}%".format((len(residus[residus<1]) / len(residus)) * 100))
    st.write("Proportion < 5% d'ecart  : {:.2f}%".format((len(residus[residus<5]) / len(residus)) * 100))
    st.write("Proportion < 10% d'ecart : {:.2f}%".format((len(residus[residus<10]) / len(residus)) * 100))

def calcul_residus(y_pred, y_test):
    residus = []
    for i in range(len(y_test)):
        residus.append(((y_pred[i] - y_test.values[i]) / y_test.values[i]) * 100)

    return np.absolute(residus)

###############################################
# Chargement et préparation du jeu de données #
###############################################

@st.cache_data
def chargement_dataset():
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

    df_carb = pd.get_dummies(df["Carburant"])
    df_carb = df_carb.replace({True:1, False:0})

    df["Carburant"] = encoder_le.fit_transform(df["Carburant"])

    X = df.drop(columns="CO2 (g/km)")
    X = scaler.fit_transform(X)
    X_ohe = pd.concat([df["Consommation mixte (l/100km)"], df_carb], axis=1)

    y = df["CO2 (g/km)"]
    y_ohe = df["CO2 (g/km)"]

    return df_original, X, y, X_ohe, y_ohe

##########################
# Chargement des modèles #
##########################

@st.cache_data
def chargement_models():

    # Chargement du modèle DecisionTree
    model_dt = load("decision_tree")

    # Chargement du réseau de neurones
    #inputs = layers.Input((2, ), name="inputs")
    #dense1 = layers.Dense(16, activation="relu", name="dense1")
    #dense4 = layers.Dense(1, name="output")

    #x=dense1(inputs)
    #outputs=dense4(x)

    #optimizer = optimizers.Adam()

    #model_dl = models.Model(inputs = inputs, outputs = outputs)
    #model_dl.load_weights("model_dl_france.h5")
    #model_dl.compile(loss="mean_squared_error", optimizer=optimizer)

    model_dl = load("model_dl_louis_new")

    # Chargement du modèle custom TensorFlow
    model_tf = load("model_tf_france")

    return model_dt, model_dl, model_tf

df_original, X, y, X_ohe, y_ohe = chargement_dataset()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=9001)
X_train_ohe, X_test_ohe, y_train_ohe, y_test_ohe = train_test_split(X_ohe, y_ohe, test_size=0.2, random_state=9001)

model_dt, model_dl, model_tf = chargement_models()

if page == pages[4] : 
    st.write("### Modélisations")

    choix = ['DecisionTree', 'Réseau de neurones'
             , 'Modèle custom TensorFlow']
    option = st.selectbox('Choix du modèle', choix)
    st.write('Le modèle choisi est :', option)

    # Prédiction avec le modèle DecisionTree
    if option == 'DecisionTree':
        y_pred = model_dt.predict(X_test) 
        residus = calcul_residus(y_pred, y_test)
        affichage_metrics(residus, y_pred, y_test) 

        fig = plt.figure()
        plt.scatter(y_pred, y_test, s=5,  c="g")
        plt.plot((0, 600), (0, 600))
        st.pyplot(fig)

    # Prédiction avec le réseau de neurones
    if option == 'Réseau de neurones':
      y_pred = model_dl.predict(X_test)
      residus = calcul_residus(y_pred, y_test)
      affichage_metrics(residus, y_pred, y_test)

      fig = plt.figure()
      plt.scatter(y_pred, y_test, s=5,  c="g")
      plt.plot((0, 600), (0, 600))
      st.pyplot(fig)

    # Prédiction avec le modèle custom TensorFlow
    if option == 'Modèle custom TensorFlow':
      y_pred = model_tf(X_test_ohe[X_test_ohe.columns[0]], X_test_ohe[X_test_ohe.columns[1]], X_test_ohe[X_test_ohe.columns[2]], X_test_ohe[X_test_ohe.columns[3]], X_test_ohe[X_test_ohe.columns[4]], X_test_ohe[X_test_ohe.columns[5]])
      residus = calcul_residus(y_pred, y_test_ohe)
      affichage_metrics(residus, y_pred, y_test_ohe)

      fig = plt.figure()
      plt.scatter(y_pred, y_test, s=5, c="g")
      plt.plot((0, 600), (0, 600))
      st.pyplot(fig)
