import pandas as pd

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
    
    # 1. Nettoyage des températures
    colonnes_temp = [colonne for colonne in df_propre.columns if colonne.startswith('T_')]
    for colonne in colonnes_temp:
        df_propre = df_propre[df_propre[colonne] >= 0]
        
    # 2. Nettoyage de l'humidité et de la hauteur
    colonnes_a_verifier_zero = ['H_data', 'AH_data'] 
    for colonne in colonnes_a_verifier_zero:
        if colonne in df_propre.columns:
            df_propre = df_propre[df_propre[colonne] > 0]
            
    # Suppression des lignes qui contiennent des cases vides
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