#!/usr/bin/env python
# coding: utf-8

# # Trousse à outils

# # Librairies

# In[ ]:


import pandas as pd
import numpy as np
import seaborn as sns

import scipy.stats as stats
from scipy.stats import shapiro
from scipy.stats import pearsonr
from scipy.stats import f_oneway
from scipy.stats import kstest
from scipy.stats import anderson
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.stats import shapiro
from scipy import stats
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.stats import t, shapiro

import statsmodels
import statsmodels.api as sm
import statsmodels.formula.api as smf

from statsmodels.formula.api import ols
from statsmodels.stats.diagnostic import het_white , normal_ad
from statsmodels.compat import lzip
from statsmodels.api import Logit
from statsmodels.genmod.generalized_linear_model import GLM
from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels.api as sm

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.model_selection import train_test_split
from sklearn.cluster import AgglomerativeClustering
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import MaxAbsScaler
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn import decomposition, preprocessing,cluster
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn import neighbors
from sklearn import decomposition
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn import decomposition, preprocessing
from sklearn import cluster, metrics
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
from sklearn.linear_model import LinearRegression
from sklearn.metrics import roc_auc_score, roc_curve, accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score




import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap

import seaborn as sns


# # Fonctions de contrôle

# In[5]:


#contrôler l'unicité d'une clé 
def controle_pk(df, col_pk):
    if len(df) != len(df[col_pk].drop_duplicates()):
        print("La clé n'est pas unique")
        print("")
        print("Les lignes concernées sont: ")
        print("")
        print(f"{df.loc[df[col_pk].duplicated()==True]}")
    else:print("La clé est unique")


# In[ ]:


#Supprimer les doublons d'une colonne
def drop_doublons(df,col,):
    x=df.loc[df[col].duplicated()==True].index
    df=df.drop(x)


# In[ ]:


def nan_control (df):
    #Controler le nombre de valeur null 
    if df[df.isna().any(axis=1)].shape[0] != 0:
        print(f"le dataframe contient {df[df.isna().any(axis=1)].shape[0]} lignes contenant des NaN")
        print(f"Cela represente {round((((df[df.isna().any(axis=1)].shape[0])/(df.shape[0]))*100),1)} % du dataframe.")
        print("")
        print(f"Les lignes concernées sont {df[df.isna().any(axis=1)].index.tolist()}")
        print("")
        print("Affichage des lignes concernées: ")
        print("")
        print(df[df.isna().any(axis=1)])
        print("")
        plt.figure(figsize=(32,20))
        sns.heatmap(df.isnull(), cbar=False, cmap="YlGnBu")
        plt.show()
    else: print("le dataframe ne contient pas de NaN ")    


# In[2]:





# In[10]:


#Donne le Z_score des données d'une colonne.
def z_score(df,col,thresold):
    score_z = np.abs(stats.zscore(df[col]))
    threshold = thresold
    z=np.where(score_z > 3)
    array = np.array(z)
    rows, columns = array.shape
    print(f"Avec la methode z_score et en choisissant un seuil de {thresold}, le nombre d'outliers est de {columns}.")


# In[11]:


#Donne le nombre d'outliers selon la methode IQR
def outliers_iqr (df,col):
    Q1 = np.percentile(df[col], 25,
                   interpolation = 'midpoint')
    Q3 = np.percentile(df[col], 75,
                   interpolation = 'midpoint')
    IQR = round((Q3 - Q1),2)
    mediane=np.percentile(df[col], 50) 
    seuil_max = round(Q3 + (1.5*IQR),2)
    outliers=df[df[col] > seuil_max]
    classement_outliers=outliers.sort_values(by=[col],ascending=False)
    
    print(f'mediane = {mediane}')
    print(f'Q1 = {Q1}')
    print(f'Q3 = {Q3}')
    print(f'IQR = {IQR}')
    print("---------------------------------------------------------------")
    print(f"Le seuil max est de {seuil_max} et le nombre d'outliers est de {outliers.shape[0]} ")
    print("---------------------------------------------------------------")
    print(f"{classement_outliers.head(outliers.shape[0])}")


# In[ ]:


#Determine le nombre d'outliers selon la methode Z_score et IQR
#Affiche le classement des outliers si ils sont presents.
def outliers_full (df,col,threshold,titre,x,y):
    score_z = np.abs(stats.zscore(df[col]))
    threshold = threshold
    z=np.where(score_z > 3)
    array = np.array(z)
    rows, columns = array.shape
    
    
    Q1 = np.percentile(df[col], 25,
                   interpolation = 'midpoint')
    Q3 = np.percentile(df[col], 75,
                   interpolation = 'midpoint')
    IQR = round((Q3 - Q1),2)
    mediane=np.percentile(df[col], 50) 
    seuil_max = round(Q3 + (1.5*IQR),2)
    outliers=df[df[col] > seuil_max]
    classement_outliers=outliers.sort_values(by=[col],ascending=False)
    
    # Ameliorer les print*
    print(f"Avec la methode z_score et en choisissant un seuil de {threshold}, le nombre d'outliers est de {columns}.")
    print("---------------------------------------------------------------------------------------------------------------")
    print(f'mediane = {mediane}')
    print(f'Q1 = {Q1}')
    print(f'Q3 = {Q3}')
    print(f'IQR = {IQR}')
    print("---------------------------------------------------------------------------------------------------------------")
    print("Analyse univarié")
    print("")
    print(f"{df[col].describe()}")
    print("---------------------------------------------------------------------------------------------------------------")
    print(f"Le seuil max est de {seuil_max} et le nombre d'outliers est de {outliers.shape[0]} ")
    
    if outliers.shape[0] > 0 :
        print("---------------------------------------------------------------------------------------------------------------")
        
        print(f"{classement_outliers.head(outliers.shape[0])}")
        print("---------------------------------------------------------------------------------------------------------------")
        print(f"{sns.boxplot(df[col])}")
        print("---------------------------------------------------------------------------------------------------------------") 
        data_graphique=classement_outliers[[x,y]].set_index(y)
        #Ameliorer le code pour obtenir directement une liste*
        print(f"liste des outliers par nom pour creation d'une liste:")
        print("")
        print(f"{classement_outliers[y].head(outliers.shape[0])}")
        print("---------------------------------------------------------------------------------------------------------------")
        data_graphique.plot.barh()
        plt.title(titre)
        plt.xlabel(x)
        plt.ylabel(y)
        plt.grid()
        plt.yticks(size=10)
        plt.show()
    else: print("")


# In[ ]:


#Retourne un df contenant les composantes principales d'un df
def composantes_mean (df,col_index,type_scaler,nb_components,col_clusters,nb_clusters):

    # Les valeurs doivent être au format array et sont enregistrées dans une variable X
    X=df.drop(columns=[col_clusters]).values
 
    
    # Le nom des colonnes sont enregistrées dans une variable features
    features=df.drop(columns=[col_clusters]).columns 
    # Les valeurs de l'index sont enregistrées dans une variable names
    names=df.drop(columns=[col_clusters]).index

    # Centrage et Reduction
    
    #On scale et on fit les données du dataframe prealablement transformées en np.array -->(X)
    scaler = type_scaler()
    scaler.fit(X)

    X_scaled = scaler.transform(X)
    
    #Verification de la moyenne et ecart type
    idx=["mean","std"]
 
    
    n_components = nb_components
  
    pca=PCA(n_components=n_components)
    pca.fit(X_scaled)
    
    #variance captée par chaque nouvelle composante 
    pca.explained_variance_ratio_
    
    #Enregistrement des variances captées dans une variable scree
    scree=(pca.explained_variance_ratio_*100).round(2)
    
    #Enregistrement de la somme cumulée des variances dans une variable
    scree_cum=scree.cumsum().round()
    
    x_list=range(1, n_components+1)
 

    pcs = pca.components_
    pcs = pd.DataFrame(pcs)
    pcs.columns = features
    pcs.index = [f"F{i}" for i in x_list]
    composantes=pcs
    
    return composantes


# # Nouvelles fonctions 

# In[ ]:


def analyse_basic(df):
    
  # Afficher les premieres du dataframe
    print("Afficher les premières lignes du dataframe:\n ")
    display(df.head(2))
    print()

  # Afficher les informations générales sur le dataframe
    print("Afficher les informations générales sur le dataframe:\n")
    display(df.info())
    print()

  # Afficher la moyenne, l'écart-type, le minimum et le maximum de chaque colonne numérique
    print("Afficher la moyenne, l'écart-type, le minimum et le maximum de chaque colonne numérique:\n")
    display(df.describe())
    print()

  # Afficher le nombre de valeurs uniques dans chaque colonne
    print("Afficher le nombre de valeurs uniques dans chaque colonne:\n")
    display(df.nunique())
    print()

  # Afficher la fréquence des valeurs dans chaque colonne
    print("Afficher la fréquence des valeurs dans chaque colonne:\n")
    display(df.value_counts())
    print()

  # Afficher le nombre de valeurs manquantes par colonne
    print("Afficher le nombre de valeurs manquantes par colonne:\n")
    display(df.isnull().sum())
    print()


# In[ ]:


def separation_nan(df):
    # Sélection des lignes sans valeurs manquantes
    df_no_nan = df.dropna()
    
    # Sélection des lignes uniquement avec valeurs manquantes
    df_nan = df[df.isna().any(axis=1)]
    
    #Affiche les 2 premieres lignes des nouveaux df
    display(df_no_nan.head(2))
    display(df_nan.head(2))     
    
    #Retourne les 2 nouveaux df
    return df_no_nan, df_nan


# In[ ]:


def mtx_corr (df):
       
       corr = df.corr()
       
       mask = np.zeros_like(corr)
       mask[np.triu_indices_from(mask)] = True
       plt.figure(figsize=(16,8))
       sns.heatmap(corr,mask=mask,center=0,cmap="coolwarm",linewidths=1,annot=True,fmt=".2f",vmin=-1,vmax=1)

       plt.title("Correlations entre les variables")

       plt.show()
   
   # Afficher les correlations positives supérieures à 0.5
       print("Correlations positives supérieures à 0.5 :")
       for i in range(len(corr.columns)):
           for j in range(i+1, len(corr.columns)):
               if corr.iloc[i, j] > 0.5:
                   print(f"{corr.columns[i]} - {corr.columns[j]}: {corr.iloc[i, j]:.2f}")

   # Afficher les correlations negatives inférieures à -0.5
       print("\nCorrelations negatives inférieures à -0.5 :")
       for i in range(len(corr.columns)):
           for j in range(i+1, len(corr.columns)):
               if corr.iloc[i, j] < -0.5:
                   print(f"{corr.columns[i]} - {corr.columns[j]}: {corr.iloc[i, j]:.2f}")


# In[ ]:


def no_bool_df(df):
    """
    Création d'un df ne contenant pas de colonnes de type bool.
    
    Parametres:
    df: le dataframe qui doit être copié
    
    Returns:
    no_bool_df: la copie du df sans les colonnes de type bool
    """
    # Selection des colonnes contenant des données de type bool
    bool_col = [col for col in df.columns if df[col].dtype == bool]
    
    # Create a copy of the dataframe that excludes the bool columns
    no_bool_df = df.drop(bool_col, axis=1)
    
    display(no_bool_df.head(1))
    
    return no_bool_df


# In[ ]:


def regression_lin_multi(df, target_col):
    # Sélectionner toutes les colonnes du dataframe excepté la colonne cible
    X = df.drop(target_col, axis=1)

    # Créer le modèle de régression linéaire multiple
    model = smf.ols(f"{target_col} ~ {' + '.join(X.columns)}", data=df)
    result = model.fit()
    print(result.summary())    

    # Afficher les colonnes dont la P-Value est supérieure au seuil alpha de 5%
    print("Colonnes avec une P-Value supérieure au seuil alpha de 5% :")
    for i, p_value in enumerate(result.pvalues):
        if p_value > 0.05:
            print(f"{result.pvalues.index[i]}: {p_value:.2f}")

    # Afficher les valeurs de R² et de R² ajusté
    print("\nValeur de R² :", result.rsquared)
    print("Valeur de R² ajusté :", result.rsquared_adj)
    if result.rsquared >= 0.5:
        print("Les valeurs de R² et de R² ajusté sont satisfaisantes.")
    else:
        print("Les valeurs de R² et de R² ajusté sont non satisfaisantes.")


# In[ ]:


def drop_outliers(df, col, threshold):
    # Calcul des scores Z pour chaque valeur de la colonne
    score_z = np.abs(stats.zscore(df[col]))
    threshold = threshold
    z=np.where(score_z > threshold)
    array = np.array(z)
    rows, columns = array.shape

    if rows == 0:
        return df
    else:
        # Suppression des lignes avec des scores Z supérieurs au seuil
        new_df = df[np.abs(df[col]) < threshold]

        return new_df


# In[ ]:


def passage_robust_scaller(df,cols_selection):
    
    #On enregistre une copy du df
    df_copy=df.copy()

    #On selectionne les colonnes que l'on passe au scaler
    df_selection_cols = df_copy.iloc[:, cols_selection ]

    #On enregistre le nom des colonnes dans une variable
    cols_name=df_selection_cols.columns

    #On applique le scaler
    transformer = RobustScaler().fit(df_selection_cols)
    df_selection_cols = transformer.transform(df_selection_cols)

    #on transforme les données en df
    df_selection_cols=pd.DataFrame(df_selection_cols,columns=cols_name)

    # Affecter les colonnes passées en RobustScaller aux colonnes correspondantes dans le dataframe d'origine
    df_copy.iloc[:, cols_selection] = df_selection_cols

    #On affiche le nouveau dataframe

    display(df_copy.head(1))
    return df_copy


# In[ ]:


def shapiro_test(data,titre):
    stat, p = stats.shapiro(data)
    print("P-value: ", p)
    if p > 0.05:
        print("On ne rejette pas l'hypothèse nulle.")
        print("La distribution suit une loi normale.")
    else:
        print("On rejette l'hypothèse nulle.")
        print("La distribution ne suit pas une loi normale.")
        
        #Representation graphique
    fig, ax = plt.subplots(figsize = (14,8))
    ax.set_title(titre,
     fontsize=22, weight='bold', color='Black', loc='center',pad=30)
    plt.box(False)
    ax.yaxis.grid(linewidth=0.5,color='grey',linestyle='-.')
    ax.xaxis.grid(linewidth=0.5,color='grey',linestyle='-.')

    sns.histplot(data, kde=True)
    plt.ylabel("Nombre", weight='bold', size=16)
    plt.xlabel("individus", weight='bold', size=16)
    plt.yticks(np.arange(0, 150, 10))
    plt.xticks(rotation=0, fontsize=12)
    plt.yticks(rotation=0, fontsize=12)
        
        
    
    plt.show()


# In[ ]:


def control_nan_column(df, column_name):
    # Nombre de valeurs NaN dans la colonne
    nan_count = df[column_name].isna().sum()

    # Pourcentage de valeurs NaN par rapport au nombre total de lignes
    nan_percent = nan_count / df.shape[0] * 100

    print('Il y a {} valeurs NaN dans la colonne {} ({:.2f}% du nombre total de lignes)'.format(nan_count, column_name, nan_percent))


# In[ ]:


def eboulis_valeurs_propre(pca):
    scree = pca.explained_variance_ratio_*100
    plt.bar(np.arange(len(scree))+1, scree)
    plt.plot(np.arange(len(scree))+1, scree.cumsum(), c="red", marker='o')
    plt.xlabel("rang de l'axe d'inertie")
    plt.ylabel("pourcentage d'inertie(%)")
    plt.title("Éboulis des valeurs propres")
    plt.show(block=False)


# In[ ]:


# Définition de la fonction pour le graphique Cercle de corrélation
def display_circles(pcs, n_comp, pca, axis_ranks, labels=None, label_rotation=0, lims=None):
    for d1, d2 in axis_ranks:  # On affiche les 3 premiers plans factoriels, donc les 6 premières composantes
        if d2 < n_comp:

            # initialisation de la figure
            fig, ax = plt.subplots(figsize=(10, 10))

            # détermination des limites du graphique
            if lims is not None:
                xmin, xmax, ymin, ymax = lims
            elif pcs.shape[1] < 30:
                xmin, xmax, ymin, ymax = -1, 1, -1, 1
            else:
                xmin, xmax, ymin, ymax = min(pcs[d1, :]), max(
                    pcs[d1, :]), min(pcs[d2, :]), max(pcs[d2, :])

            # affichage des flèches
            # s'il y a plus de 30 flèches, on n'affiche pas le triangle à leur extrémité
            if pcs.shape[1] < 30:
                plt.quiver(np.zeros(pcs.shape[1]), np.zeros(pcs.shape[1]),
                           pcs[d1, :], pcs[d2, :],
                           angles='xy', scale_units='xy', scale=1, color="grey")
            else:
                lines = [[[0, 0], [x, y]] for x, y in pcs[[d1, d2]].T]
                ax.add_collection(LineCollection(
                    lines, axes=ax, alpha=.1, color='black'))

            # affichage des noms des variables
            if labels is not None:
                for i, (x, y) in enumerate(pcs[[d1, d2]].T):
                    if x >= xmin and x <= xmax and y >= ymin and y <= ymax:
                        plt.text(x, y, labels[i], fontsize='14', ha='center',
                                 va='center', rotation=label_rotation, color="blue", alpha=0.5)

            # affichage du cercle
            circle = plt.Circle((0, 0), 1, facecolor='none', edgecolor='b')
            plt.gca().add_artist(circle)

            # définition des limites du graphique
            plt.xlim(xmin, xmax)
            plt.ylim(ymin, ymax)

            # affichage des lignes horizontales et verticales
            plt.plot([-1, 1], [0, 0], color='grey', ls='--')
            plt.plot([0, 0], [-1, 1], color='grey', ls='--')

            # nom des axes, avec le pourcentage d'inertie expliqué
            plt.xlabel('F{} ({}%)'.format(
                d1+1, round(100*pca.explained_variance_ratio_[d1], 1)))
            plt.ylabel('F{} ({}%)'.format(
                d2+1, round(100*pca.explained_variance_ratio_[d2], 1)))

            plt.title("Cercle des corrélations (F{} et F{})".format(d1+1, d2+1))
            plt.show(block=False)


# In[ ]:


def verif_billet_rl(csv):
    billet_test= pd.read_csv(csv)
    billet_value=billet_test.drop('id', axis=1)
    y_pred = model_logit.predict(billet_value)
    proba_true = model_logit.predict_proba(billet_value)[:, 1]
    billet_test['Prediction'] = y_pred
    billet_test['Probability_is_true'] = proba_true.round(3)
    billets_predict_rl = billet_test[['id','Prediction','Probability_is_true']].set_index("id")
    return billets_predict_rl
