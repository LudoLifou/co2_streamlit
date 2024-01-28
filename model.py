import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Definition des chemins et noms 
images_path = './data/images/'     


# Import d'un modèle entrainé
# Import d'un X_train, y_train
# Entrer un nouveau vehicule en choisissant les parametres --> Prédiction
# Choix d'un véhicule dans la base de donnée, (propositions a partir de criteres fixés, ou véhicule "le plus proche") 
# --> Prédiction / vrai valeur
# Peut on implementer un predictif partiel, a partir de quelques (ou une) caractéristiques. --> Donne le Co2 ou son intervalle ou la proba de classe (plutot stat alors)
# Montrer les quelques graphiques réalisés (quelle interactivité ? plutot base de donnée d'images)


# Page "Model"

def app(df) :
    st.write("# Algorithmes de Machine Learning")

    st.write("""
            Nous avons expérimentés plusieurs modèles, le principe étant:  
            1. De tester différents algorithmes et méthodes en augmentant progressivement leur complexité, pour comparer leurs performances (scores, robustesse, vitesse). &nbsp;&nbsp; => &nbsp;&nbsp;    Selection du ou des meilleurs modèles  
            2. De chercher à améliorer le (ou les) modèle final retenu (optimisation des paramètres)  
            3. Enfin, de s'interesser à l'interprétabilité et la  "feature importance" de ces modèles finaux, pour analyse.
             
            Nous avons mené cette démarche à la fois pour la **régression** et la **classification**. 
            """)



    tab1, tab2, Tab3 = st.tabs(["regression", "classification", "deep learning"])

    with tab1:
            
        st.write("""
                 # Problème de régression
                 ---
                 ### 1. Modèle de Regression linéaire 
                 - ###### Modèle simple sans régularisation
                 - ###### Modèle régularisé avec recherche des meilleurs paramètres.
                   Même si notre premier modèle ne semblait pas conduire à du surapprentissage, nous avons voulu tester differents paramètres de régularisation \
                  afin de le confirmer, et observer les effets des régularisations (Lasso, Ridge).  
                 
                   **La régularisation n'apporte rien !!**  
                   Les meilleurs paramères trouvés par grille de recherche et valdation croisée sont pour alpha = 0, c'est a dire sans régularisation.   
                 """)
        
        col1, col2, clo3 = st.columns([0.25, 0.45, 0.3], gap = 'medium')
        with col2:
            img_name = "lr_elastic_net.png"    
            st.image(images_path + img_name,
            use_column_width= True )

        st.write("""
                    ##### Résultats :  """)
        col1, col2 = st.columns([0.7, 0.3])
        with col1:
            st.write("""   
                    Les scores obtenus sur l'échantillon d'apprentissage et l'échantillon de test sont très proches et ne montrent pas de sur-apprentissage. Ce que confirme une validation croisée.  
                    L'erreur moyenne sur les prédictions est d'environ 10g/km pour des valeurs de l'ordre de 130 g/km.  
                    L'erreur moyenne relative est d'environ 10%  
                    """)
        with col2:
            img_name = "score_lr.png"
            img = "./data/images/lr_elastic_net.png"                 
            st.image(images_path + img_name,
            use_column_width= True )

        st.write("Ce modèle donne déjà des résultats qui semblent corrects avec un R2_score de 0,89, c’est-à-dire que 89% de la variance du CO2 peut être expliquée par ce modèle de régression.")





        """
                    use_column_width= ",
        o	Graphique
        o	Pas d'overfitting
        -	Score rapide, conclusion sur les features (mini graph ?)

        """



                  
                  

          


    # tab2.subheader("A tab with the data")
    # tab2.write(data)




    # def lr(Engine_Size, Cylinders, Fuel_Consumption_City,Fuel_Consumption_Hwy, Fuel_Consumption_Comb,Fuel_Consumption_Comb_mpg,Fuel_Type_E, Fuel_Type_X,Fuel_Type_Z, Transmission_A4, Transmission_A5, Transmission_A6,Transmission_A7, Transmission_A8, Transmission_A9,Transmission_AM5, Transmission_AM6, Transmission_AM7,Transmission_AM8, Transmission_AM9, Transmission_AS10,Transmission_AS4, Transmission_AS5, Transmission_AS6,Transmission_AS7, Transmission_AS8, Transmission_AS9,Transmission_AV, Transmission_AV10, Transmission_AV6,Transmission_AV7, Transmission_AV8, Transmission_M5,Transmission_M6, Transmission_M7, Make_Type_Luxury,Make_Type_Premium, Make_Type_Sports, Vehicle_Class_Type_SUV,Vehicle_Class_Type_Sedan, Vehicle_Class_Type_Truck):
    # c=pd.DataFrame([Engine_Size, Cylinders, Fuel_Consumption_City,Fuel_Consumption_Hwy, Fuel_Consumption_Comb,Fuel_Consumption_Comb_mpg,Fuel_Type_E, Fuel_Type_X,Fuel_Type_Z, Transmission_A4, Transmission_A5, Transmission_A6,Transmission_A7, Transmission_A8, Transmission_A9,Transmission_AM5, Transmission_AM6, Transmission_AM7,Transmission_AM8, Transmission_AM9, Transmission_AS10,Transmission_AS4, Transmission_AS5, Transmission_AS6,Transmission_AS7, Transmission_AS8, Transmission_AS9,Transmission_AV, Transmission_AV10, Transmission_AV6,Transmission_AV7, Transmission_AV8, Transmission_M5,Transmission_M6, Transmission_M7, Make_Type_Luxury,Make_Type_Premium, Make_Type_Sports, Vehicle_Class_Type_SUV,Vehicle_Class_Type_Sedan, Vehicle_Class_Type_Truck]).T
    # return model.predict(c)
          
    
    # prediction=lr(Engine_Size, Cylinders, Fuel_Consumption_City,Fuel_Consumption_Hwy, Fuel_Consumption_Comb,Fuel_Consumption_Comb_mpg,Fuel_Type_E, Fuel_Type_X,Fuel_Type_Z, Transmission_A4, Transmission_A5, Transmission_A6,Transmission_A7, Transmission_A8, Transmission_A9,Transmission_AM5, Transmission_AM6, Transmission_AM7,Transmission_AM8, Transmission_AM9, Transmission_AS10,Transmission_AS4, Transmission_AS5, Transmission_AS6,Transmission_AS7, Transmission_AS8, Transmission_AS9,Transmission_AV, Transmission_AV10, Transmission_AV6,Transmission_AV7, Transmission_AV8, Transmission_M5,Transmission_M6, Transmission_M7, Make_Type_Luxury,Make_Type_Premium, Make_Type_Sports, Vehicle_Class_Type_SUV,Vehicle_Class_Type_Sedan, Vehicle_Class_Type_Truck)
    # return render_template('index.html',prediction_text="Co2 Emissions by car is {}".format(np.round(prediction,2)))

# # Inserer dans fichier d'aide: Remplir les valeurs manquante, méthode rapide
#     for col in X_cat.columns:   
#         X_cat[col] = X_cat[col].fillna(X_cat[col].mode()[0])
#     for col in X_num.columns:
#         X_num[col] = X_num[col].fillna(X_num[col].median())
#     X_cat_scaled = pd.get_dummies(X_cat, columns=X_cat.columns)
#     X = pd.concat([X_cat_scaled, X_num], axis = 1)

#     from sklearn.model_selection import train_test_split
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

#     from sklearn.preprocessing import StandardScaler
#     scaler = StandardScaler()
#     X_train[X_num.columns] = scaler.fit_transform(X_train[X_num.columns])
#     X_test[X_num.columns] = scaler.transform(X_test[X_num.columns])

# # Inserer dans fichier d'aide: tester plusieurs clf, méthode rapide
#     from sklearn.ensemble import RandomForestClassifier
#     from sklearn.svm import SVC
#     from sklearn.linear_model import LogisticRegression
#     from sklearn.metrics import confusion_matrix

#     def prediction(classifier):
#         if classifier == 'Random Forest':
#             clf = RandomForestClassifier()
#         elif classifier == 'SVC':
#             clf = SVC()
#         elif classifier == 'Logistic Regression':
#             clf = LogisticRegression()
#         clf.fit(X_train, y_train)
#         return clf

#     def scores(clf, choice):
#         if choice == 'Accuracy':
#             return clf.score(X_test, y_test)
#         elif choice == 'Confusion matrix':
#             return confusion_matrix(y_test, clf.predict(X_test))
        
#     choix = ['Random Forest', 'SVC', 'Logistic Regression']
#     option = st.selectbox('Choix du modèle', choix)
#     st.write('Le modèle choisi est :', option)

#     clf = prediction(option)
#     display = st.radio('Que souhaitez-vous montrer ?', ('Accuracy', 'Confusion matrix'))
#     if display == 'Accuracy':
#         st.write(scores(clf, display))
#     elif display == 'Confusion matrix':
#         st.dataframe(scores(clf, display)