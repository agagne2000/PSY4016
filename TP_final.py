import pandas as pd 
import seaborn as sb
import scipy as sp 
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import sqlite3 as sql
import sklearn
from sklearn import model_selection
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture as GMM

fichier_excel = "AliceGagné_données_psy4016-H22_20220211_Immunotherapy.xlsx"            #ex. pipeline : fichier py facilement modifiable 

# ----------------------------------------------------------------------
class Sauvegarder_donnees:          #classe servant à l'enregistrement des données
    def __init__(self, base_de_donnees, fichier, df):           #initialisation des variables de la classse 
        self.base_de_donnees = base_de_donnees
        self.fichier = fichier
        self.df = df
        self.conn = sql.connect(self.base_de_donnees)

    def creer_fichier(self):            #fonction qui crée le fichier où les données seront enregistrées
        self.table = f""" CREATE TABLE IF NOT EXISTS {self.fichier} (
                        sex CHAR(10),
                        age CHAR(10),
                        Time CHAR(10),
                        Nb_warts CHAR(10),
                        Type CHAR(10),
                        Area CHAR(10),
                        Induration CHAR(10),
                        Result CHAR(10)
                        ); """
        self.conn.execute(self.table)

    def enregistrer_donnees(self):          #fonction pour l'enregistrement des données
        self.commande = """ INSERT INTO Info(sex, age, Time, Nb_warts, Type, Area, Induration, Result) VALUES(?,?,?,?,?,?,?,?);"""

        self.str10 = lambda x : str(x).ljust(10, ' ')           #fonction anonyme (lambda)

        try:            #script pour la gestion des erreurs
            self.cur = self.conn.cursor()

            for i in self.df.index:
                self.data_tuple = (self.str10(self.df.sex[i]), self.str10(self.df.age[i]), self.str10(self.df.Time[i]), 
                                    self.str10(self.df.Number_of_Warts[i]), self.str10(self.df.Type[i]), self.str10(self.df.Area[i]), 
                                    self.str10(self.df.induration_diameter[i]), self.str10(self.df.Result_of_Treatment[i]))
                self.cur.execute(self.commande, self.data_tuple)
        except Exception as e:
            print(f"Erreur lors de l'enregistrement des données... : {e}")
        else:
            print("Étape d'enregistrement des données contenant script pour gestion des erreurs terminée.")
        finally:
            print("Fin de l'enregistrement des données.")
        
        self.conn.commit()

    def fermer_base_de_donnees(self):           #fonction qui ferme le fichier contenant toutes les données 
        self.conn.close()
# ----------------------------------------------------------------------

# ----------------------------------------------------------------------
def recherche_lineaire(valeur, table):          #algorithme d'automatisation (recherche linéaire)
    reponse = False
    
    for i in table: 
        
        if i == valeur:
        
            reponse = True
            
    return reponse
# ----------------------------------------------------------------------

# ----------------------------------------------------------------------
def nettoyage_donnees():            #fonction qui sert à nettoyer les données 
    df = pd.read_excel(fichier_excel)
           
    for colonne in df.columns:          #correction des erreurs dans les données
        for ligne in df.index:
            if colonne == "sex":
                if df.at[ligne, colonne] == "M":
                    df.at[ligne, colonne] = 1
            if colonne == "age":
                if df.at[ligne, colonne] == "36y":
                    df.at[ligne, colonne] = 36

    age_plus_frequent = df["age"].mode()            #script pour considérer les valeurs manquantes (remplacement par âge le plus fréquent)
    df["age"] = df["age"].fillna(int(age_plus_frequent))

    #pandas a transformé automatiquement les "." du fichier excel en "," et les float en int

    sbd = Sauvegarder_donnees("Donnees.sqlite", "Info", df)         #enregistrement dans la base de données sqlite 
    sbd.creer_fichier()
    sbd.enregistrer_donnees()
    sbd.fermer_base_de_donnees()

    st.write("Algorithme d'automatisation")
    st.write("Recherche linéaire: " + str(recherche_lineaire(20, df["age"].values)))            #appel de la fonction recherche_lineaire
    st.write("---------------------")
# ----------------------------------------------------------------------

# ----------------------------------------------------------------------
def correlation():          #fonction qui calcule la matrice de corrélations de pearson et crée le corrélogramme 
    st.write("Objectif #1")
    df1 = pd.read_excel(fichier_excel,          #nouveau df avec les variables d'intérêt
                        usecols = ["Time", "Number_of_Warts", "Area", "induration_diameter"])
    df1.columns = ["Temps", "Nombre", "Surface", "Diamètre"]

    DF1 = df1.corr(method="pearson")            #calcul de la matrice de corrélations

    st.write("Matrice de corrélations de Pearson : ")
    st.dataframe(DF1)

    fig1, ax = plt.subplots()           #création du corrélogramme 
    sb.heatmap(DF1, vmin=-1, vmax=1, cmap="GnBu",annot=True, fmt='.2g', linewidths=0.2, linecolor='white', 
                cbar=True, cbar_kws=None, cbar_ax=None, square=True, xticklabels='auto', yticklabels='auto',alpha=0.5)

    st.write("Corrélogramme : ")
    st.pyplot(fig1) 
    st.write("---------------------")          
# ----------------------------------------------------------------------

# ----------------------------------------------------------------------
def chi2():         #fonction qui calcule le chi2 et l'histogramme de catégories 
    st.write("Objectif #2")
    df2 = pd.read_excel(fichier_excel, usecols = ["Type", "Result_of_Treatment"])           #nouveau df avec les variables d'intérêt           
    df2.columns = ["Type", "Resultat"]
    df2["Type"] = df2["Type"].apply(str)

    for colonne in df2.columns:         #changement des valeurs numériques (int) dans le df pour leur nom spécifique (str)
        for ligne in df2.index:
            if df2.at[ligne, "Type"] == "1":
                df2.at[ligne, "Type"] = "Commune"
            if df2.at[ligne, "Type"] == "2":
                df2.at[ligne, "Type"] = "Plantaire"
            if df2.at[ligne, "Type"] == "3":
                df2.at[ligne, "Type"] = "Commune et Plantaire"

    DF2 = pd.crosstab(df2["Type"], df2["Resultat"])         #calcul de la table de contingence nécessaire au calcul du chi2

    st.write("Table de contingence : ")
    st.dataframe(DF2)

    st.write("Chi2 : ")
    chi2, p, dof, expctd = sp.stats.contingency.chi2_contingency(DF2)           #calcul du chi2
    st.write(f"chi2 = {chi2},     p = {p}")         #formatage des chaînes 

    fig2, ax = plt.subplots()               #création de l'histogramme de catégories 

    fig2 = sb.catplot(x="Type", hue="Resultat", data=df2, kind="count", color="m", palette="GnBu", saturation=0.5, legend=False)
    fig2.set(xlabel = "Type de verrue", ylabel = "Nombre de personne")
    plt.legend(labels=["Non réussi", "Réussi"], title = "Résultat du traitement", loc=2)

    st.write("Histogramme de catégories : ")
    st.pyplot(fig2)
    st.write("---------------------")    
# ----------------------------------------------------------------------

# ----------------------------------------------------------------------
def aa_supervise():         #fonction pour le sript d'apprentissage automatique supervisé
    st.write("AA supervisé")
    df3 = pd.read_excel(fichier_excel,          #nouveau df avec les variables d'intérêt 
        usecols = ["Area", "Number_of_Warts"])
    df3.columns = ["Surface", "Nombre"]

    fig3, ax = plt.subplots()

    x = df3.Nombre.values           #valeurs associées à x et y 
    y = df3.Surface.values

    ax.scatter(x, y, c = "#8BB4C9")             #nuage de points contenant les données de départ
    st.write("Nuage de points sans AA supervisé : ")
    st.pyplot(fig3)

    avgx = 0.0          #calcul des moyennes 
    avgy = 0.0

    for i in df3.index:
        avgx += df3.Surface[i]/len(df3)-1
        avgy += df3.Nombre[i]/len(df3)-1
        
    totalxx = 0         #calcul des sommes 
    totalxy = 0

    for ii in df3.index:
        totalxx += (df3.Surface[ii]-avgx)**2
        totalxy += (df3.Surface[ii]-avgx)*(df3.Nombre[ii]-avgy)
        
    m = totalxy/totalxx         #calcul de la pente 
    b = avgy-m*avgx             #calcul de la constante 

    model = LinearRegression(fit_intercept=True)            #méthode utilisée = régression linéaire (données quantitatives), hyperparamètre fit

    #x.shape             #organisation des données (matrice d'entités et vecteur cible)
    X = x[:, np.newaxis]
    #X.shape
    #y.shape

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.6, random_state=0)            #validation croisée (testset validation)
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    mean_squared_error(y_test, pred)

    model.fit(X, y)             #adaptation du modèle aux données
    #model.coef_
    #model.intercept_

    xfit = np.linspace(0, 800)              #application du modèle 
    Xfit = xfit[:, np.newaxis]
    yfit = model.predict(Xfit)

    fig4, ax = plt.subplots()               #affichage de la régression linéaire dans un nuage de points 
    ax.scatter(x, y, c = "#8BB4C9")
    ax.plot(xfit, yfit, c = "#00466A")
    st.write("Nuage de points avec AA supervisé (Régression linéaire) : ")
    st.pyplot(fig4)
    st.write("---------------------")
# ----------------------------------------------------------------------

# ----------------------------------------------------------------------
def aa_non_supervise():         #fonction pour le sript d'apprentissage automatique non-supervisé
    st.write("AA non-supervisé")
    df4 = pd.read_excel(fichier_excel,
                        usecols = ["Number_of_Warts", "Type", "Area", "induration_diameter"])           #nouveau df avec les variables d'intérêt

    X_df4 = df4.drop("Type", axis=1)                #valeurs associées à x et y 
    y_df4 = df4["Type"]

    model = PCA(n_components=2)             #méthode utilisée = analyse en composantes principales (ACP)

    model.fit(X_df4)

    X_2D = model.transform(X_df4)

    df4['PCA1'] = X_2D[:, 0]
    df4['PCA2'] = X_2D[:, 1]

    fig5, ax = plt.subplots()           #nuage de points avec ACP
    fig5 = sb.lmplot(x = "PCA1", y = "PCA2", hue="Type", palette="GnBu", data=df4, fit_reg=False)
    st.write("Nuage de points avec ACP : ")
    st.pyplot(fig5)

    model = GMM(n_components=3, covariance_type='full')             #méthode plus puissante = modèle de mélange gaussien (GMM)
    model.fit(X_df4)
    y_gmm = model.predict(X_df4)
    df4['cluster'] = y_gmm

    fig6, ax = plt.subplots()               #nuage de points avec GMM
    fig6 = sb.lmplot(x = "PCA1", y = "PCA2", data=df4, hue="Type",col='cluster', palette="GnBu", fit_reg=False)
    st.write("Nuage de points avec GMM : ")
    st.pyplot(fig6)

    pca = PCA(2)            #script pour visualisation ACP (réduction de la dimentionnalité)
    projected = pca.fit_transform(df4)
    #df4.shape
    #projected.shape

    fig7, ax = plt.subplots()           #nuage de point démontrant les 2 premières composantes principales de chaque point 
    plt.scatter(projected[:, 0], projected[:, 1],
                c=df4.Type, edgecolor='none', alpha=0.5,
                cmap=plt.cm.get_cmap('Spectral', 10))
    plt.xlabel('component 1')
    plt.ylabel('component 2')
    plt.colorbar()
    st.write("Nuage de point avec visualisation ACP : ")
    st.pyplot(fig7)
    st.write("---------------------")
# ----------------------------------------------------------------------

# Pour exécuter le programme utiliser la commande suivante: python -m streamlit run TP_final.py
if __name__ == "__main__":
    st.title("Projet final - Rapport")

    nettoyage_donnees()
    correlation()
    chi2()
    aa_supervise()
    aa_non_supervise()

    # Certaines étapes sont en commentaires pour éviter que les valeurs apparaissent dans le rapports, 
    # cependant elles ont été utiles pour la réalisation des sections 