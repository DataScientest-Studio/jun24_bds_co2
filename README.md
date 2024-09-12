Machine Learning: Emission de CO2 par les véhicules
==============================

## Présentation

Ce répertoire contient le code de notre projet développé pendant notre [formation de Data Scientist](https://datascientest.com/en/data-scientist-course) avec [DataScientest](https://datascientest.com/) sous `Python 3.11.9`.

L’objectif principal de ce projet est de développer un modèle de prédiction de l'émission de CO2 des véhicules 
En utilisant les techniques de Machine Learning et Deep Learning, nous cherchons à créer un modèle de régression capable de prédire l'émission de CO2 de véhicules. Les objectifs spécifiques incluent l’acquisition et la préparation d’un ensemble de données diversifié et annoté, la conception et l'entraînement de modèles de Machine Learning et deDeep Learning robuste, ainsi que l’évaluation rigoureuse de ses performances en termes de sensibilité, de spécificité et de précision.

Ce projet a été développé par l'équipe suivante :
- Marc BASSELIER ()
- Thierry GONCALVES-NOVO ([GitHub](https://github.com/ThGoncal/) / [LinkedIn](https://www.linkedin.com/in/thierry-goncalves-novo/))
- Tanguy LALOUELLE ()
- Louise RIGAL ()

Sous le mentorat de Raphaël et Gaspard de DataScientest().

Le rapport complet de notre projet est disponible dans le dossier ([report](./report)).

Voici le résumé :

> Le transport routier contribue à environ un cinquième des émissions totales de l'Union européenne (UE) de dioxyde de carbone (CO2), le principal gaz à effet de serre (GES), dont 75 % proviennent des voitures particulières. Le secteur des transports est le seul secteur majeur de l'UE où les émissions de GES continuent d'augmenter.
> Les émissions de CO2 des voitures de tourisme sont mesurées dans le cadre du test de certification des véhicules, qui est basé sur le nouveau cycle de conduite européen (NEDC), et est également appelé test NEDC. 
> La consommation de carburant des véhicules est directement dérivée de la mesure des émissions de dioxyde de carbone (CO2), d'hydrocarbures (HC) et d'oxyde de carbone (CO) effectuées lors des tests de certification, en tenant compte du bilan carbone des gaz d'échappement. Les véhicules modernes conformes aux normes européennes (Euro5 et Euro6) ont des niveaux d'émissions de CO et de HC faibles à l'échappement (contribuant à environ 1 % de la consommation de carburant). En d'autres termes, les émissions de CO2 peuvent être considérées comme proportionnelles au carburant consommé pendant le fonctionnement du véhicule.
> L'écart croissant entre la consommation de carburant en conditions réelles et celle des véhicules homologués, ainsi que la difficulté d'évaluer l'effet réel des technologies de réduction du CO2, ont conduit l'UE à revoir la procédure d'homologation des voitures particulières et des véhicules utilitaires légers, ce qui a abouti à l'introduction de la nouvelle procédure d'essai des véhicules utilitaires légers harmonisée à l'échelle mondiale (WLTP). Cette nouvelle procédure est utilisée pour l'évaluation des émissions, y compris le CO2, dans le cadre de la réception par type des véhicules utilitaires légers depuis le 1er septembre 2017. Toutefois, les objectifs en matière de CO2 continuent d'être évalués par rapport aux valeurs de CO2 de la NEDC.
> L’Union Européenne a fixé à l’ensemble des constructeurs automobiles pour objectif une réduction des émissions moyennes de CO2 concernant l’immatriculation des voitures neuves. À partir de 2035, toutes les nouvelles voitures qui arriveront sur le marché de l'UE devraient être à zéro émission de CO2. Ces règles n'affectent pas les voitures existantes. 
>Du fait du changement de procédure d’essai à partir du 1er septembre 2017, une période de transition a été tolérée pour le passage progressivement du test NEDC au test WLTP.
Des objectifs intermédiaires ont été fixées :


# Organisation du répertoire    

Ce répertoire suit une hiérarchie classique et facile à explorer.


    ├── LICENSE
    ├── README.md          <- Le fichier README.md de niveau supérieur pour les développeurs du projet
    ├── data               <- Doit être sur votre ordinateur mais pas sur Github (uniquement en .gitignore)
    │   ├── processed      <- Les ensembles de données canoniques finaux pour la modélisation.
    │   └── raw            <- Les données brutes non transformées.
    │
    ├── models             <- Modèles entraînés et sérialisés, prédictions de modèles ou résumés de modèles
    │
    ├── notebooks          <- Jupyter notebooks. la convention de dénommination est un nombre (pour le classement),
    │                         Le nom du créateur, et une courte description délimitée par un `-`, par exemple
    │                         `1.0-alban-data-exploration`.
    │
    ├── references         <- Manuels, liens et tout autre matériel explicatif
    │
    ├── reports            <- Les rapports du projet au format PDF
    │   └── figures        <- Figures générées pour les rapports
    │
    ├── requirements.txt   <- Le fichier d'exigences pour reproduire l'environnement de développment, par exemple
    │                         généré par `pip freeze > requirements.txt`
    │
    ├── src                <- Code source à utiliser dans ce projet
    │   ├── __init__.py    <- Transfomre le code source en un module Python
    │   │
    │   ├── features       <- Scripts pour transformer les données brutes avant modélisation (Préprocessing et featuring)
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts pour entraîner des modèles, puis utiliser des modèles entraînés 
    │   │   │                 pour faire des prédictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   ├── visualization  <- Scripts pour créer des visualisations exploratoires et orientées résultats
    │   │   └── visualize.py

--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
