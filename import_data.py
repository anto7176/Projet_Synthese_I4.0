import pandas as pd
import numpy as np

def charger_donnees(chemin_x: str, chemin_y: str):
    try:
        donnees_x = pd.read_csv(chemin_x)
        print(f"Succès : {chemin_x} a été importé avec {len(donnees_x)} lignes.")
    except FileNotFoundError:
        print(f"Erreur : Le fichier {chemin_x} est introuvable.")
        donnees_x = None

    try:
        donnees_y = pd.read_csv(chemin_y)
        print(f"Succès : {chemin_y} a été importé avec {len(donnees_y)} lignes.")
    except FileNotFoundError:
        print(f"Erreur : Le fichier {chemin_y} est introuvable.")
        donnees_y = None
        
    return donnees_x, donnees_y


def nettoyer_donnees(df):
    if df is None:
        return None
        
    print(f"--- Début du nettoyage ---")
    print(f"Lignes avant nettoyage : {len(df)}")
    
    df_propre = df.copy()
    
    # 1. Correction des capteurs isolés défaillants par chambre
    # Si un capteur a une différence > 100 avec ses DEUX voisins, 
    # on le remplace par la moyenne de ses voisins pour sauver la ligne.
    for i in range(1, 6):
        c1 = f'T_data_{i}_1'
        c2 = f'T_data_{i}_2'
        c3 = f'T_data_{i}_3'
        
        # Identification des anomalies
        cond_c1 = (abs(df_propre[c1] - df_propre[c2]) > 50) & (abs(df_propre[c1] - df_propre[c3]) > 50)
        cond_c2 = (abs(df_propre[c2] - df_propre[c1]) > 50) & (abs(df_propre[c2] - df_propre[c3]) > 50)
        cond_c3 = (abs(df_propre[c3] - df_propre[c1]) > 50) & (abs(df_propre[c3] - df_propre[c2]) > 50)
        
        # Remplacement par la moyenne des voisins (division entière pour rester en int64)
        df_propre.loc[cond_c1, c1] = (df_propre.loc[cond_c1, c2] + df_propre.loc[cond_c1, c3]) // 2
        df_propre.loc[cond_c2, c2] = (df_propre.loc[cond_c2, c1] + df_propre.loc[cond_c2, c3]) // 2
        df_propre.loc[cond_c3, c3] = (df_propre.loc[cond_c3, c1] + df_propre.loc[cond_c3, c2]) // 2

    # 2. Nettoyage des températures extrêmes restantes
    # Utilisation de .clip() au lieu de .between() pour ne pas supprimer la ligne
    colonnes_temp = [colonne for colonne in df_propre.columns if colonne.startswith('T_')]
    for colonne in colonnes_temp:
        df_propre[colonne] = df_propre[colonne].clip(lower=0, upper=1200)
            
    # 3. Suppression des lignes qui contiennent de vraies cases vides d'origine (NaN)
    df_propre = df_propre.dropna()
            
    print(f"Lignes après nettoyage : {len(df_propre)}")
    print(f"Lignes supprimées : {len(df) - len(df_propre)}")
    print(f"--------------------------\n")
    
    return df_propre

def formater_index_temporel(df, nom_colonne):
    if df is None:
        return None
    
    # Conversion de la colonne en objets datetime
    df[nom_colonne] = pd.to_datetime(df[nom_colonne])
    
    df = df.set_index(nom_colonne)
    df = df.sort_index()
    
    return df


if __name__ == "__main__":
    # Chargement
    df_x, df_y = charger_donnees("data/data_X.csv", "data/data_Y.csv")
    
    # Nettoyage et Formatage
    if df_x is not None and df_y is not None:
        # Nettoyage de X
        df_x_propre = nettoyer_donnees(df_x)
        
        # Passage en DatetimeIndex
        df_x_final = formater_index_temporel(df_x_propre, "date_time")
        df_y_final = formater_index_temporel(df_y, "date_time")
        
        print("Formatage DatetimeIndex terminé pour les deux datasets.")
        # print(df_x_final.index) # Pour vérifier le format de l'index
        print(df_x_final.head(10))
        print("\nStatistiques après nettoyage (pour vérifier que les extrêmes ont disparu) :")
        print(df_x_final.describe().T[['min', 'max']])