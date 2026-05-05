import os
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

    for i in range(1, 6):
        c1 = f'T_data_{i}_1'
        c2 = f'T_data_{i}_2'
        c3 = f'T_data_{i}_3'

        cond_c1 = (abs(df_propre[c1] - df_propre[c2]) > 50) & (abs(df_propre[c1] - df_propre[c3]) > 50)
        cond_c2 = (abs(df_propre[c2] - df_propre[c1]) > 50) & (abs(df_propre[c2] - df_propre[c3]) > 50)
        cond_c3 = (abs(df_propre[c3] - df_propre[c1]) > 50) & (abs(df_propre[c3] - df_propre[c2]) > 50)

        df_propre.loc[cond_c1, c1] = (df_propre.loc[cond_c1, c2] + df_propre.loc[cond_c1, c3]) // 2
        df_propre.loc[cond_c2, c2] = (df_propre.loc[cond_c2, c1] + df_propre.loc[cond_c2, c3]) // 2
        df_propre.loc[cond_c3, c3] = (df_propre.loc[cond_c3, c1] + df_propre.loc[cond_c3, c2]) // 2

    colonnes_temp = [colonne for colonne in df_propre.columns if colonne.startswith('T_')]
    for colonne in colonnes_temp:
        df_propre[colonne] = df_propre[colonne].clip(lower=0, upper=1200)

    df_propre = df_propre.dropna()
    print(f"Lignes après nettoyage : {len(df_propre)}")
    print(f"Lignes supprimées : {len(df) - len(df_propre)}\n")

    return df_propre


def normaliser_qualite(df_y, colonne='quality'):
    if df_y is None:
        return None
    df_norm = df_y.copy()
    q_min = df_norm[colonne].min()
    q_max = df_norm[colonne].max()
    df_norm[colonne] = (df_norm[colonne] - q_min) / (q_max - q_min) * 100
    return df_norm


def formater_index_temporel(df, nom_colonne):
    if df is None:
        return None
    df[nom_colonne] = pd.to_datetime(df[nom_colonne])
    df = df.set_index(nom_colonne)
    df = df.sort_index()
    return df


def creer_process_matrix(df_x_final, df_y_final):
    """Effectue la création de la matrice pivot et la jointure (traitement long)"""
    print("⏳ Construction de la process matrix (pivot)...")
    
    df_y_decale = df_y_final.copy()
    df_y_decale.index = df_y_decale.index - pd.Timedelta(hours=1)

    VARIABLES = [
        'T_data_3_3', 'T_data_3_1', 'T_data_3_2',
        'H_data', 'T_data_5_2', 'T_data_5_1', 'T_data_5_3',
        'T_data_1_3', 'T_data_1_2', 'T_data_1_1', 'T_data_2_2', 'T_data_2_3'
    ]

    df_x_pm = df_x_final[VARIABLES].copy()
    df_x_pm['_heure']  = df_x_final.index.floor('h')
    df_x_pm['_minute'] = df_x_final.index.minute
    df_x_pm = df_x_pm[df_x_pm['_minute'] < 60]

    df_pivot = df_x_pm.pivot_table(
        index='_heure', columns='_minute',
        values=VARIABLES, aggfunc='first'
    )
    df_pivot.columns = [f"{var}_m{int(m):02d}" for var, m in df_pivot.columns]
    df_pivot.index.name = 'date_time'

    df_y_tronque = df_y_decale.copy()
    df_y_tronque.index = (df_y_tronque.index - pd.Timedelta(minutes=5)).floor('h')

    df_final = df_pivot.join(df_y_tronque, how='inner').dropna()
    print(f"Process matrix terminée : {df_final.shape[0]} lignes × {df_final.shape[1]-1} features\n")
    return df_final


def get_or_create_formatted_data():
    # Le fichier qui contiendra la DB finale complétement formatée
    path_matrix = "data/data_X_formatee.pkl"

    if os.path.exists(path_matrix):
        print("Matrice formatée existante trouvée. Chargement instantané...")
        df_final = pd.read_pickle(path_matrix)
    else:
        print("Fichier introuvable. Nettoyage et formatage de la matrice en cours...")
        df_x, df_y = charger_donnees("data/data_X.csv", "data/data_Y.csv")

        if df_x is not None and df_y is not None:
            df_x_propre = nettoyer_donnees(df_x)
            df_y_norm = normaliser_qualite(df_y, colonne='quality')

            df_x_temp = formater_index_temporel(df_x_propre, "date_time")
            df_y_temp = formater_index_temporel(df_y_norm, "date_time")

            # Traitement long (Pivot + Jointure)
            df_final = creer_process_matrix(df_x_temp, df_y_temp)

            # Sauvegarde pour ne plus jamais refaire ce calcul
            os.makedirs("data", exist_ok=True)
            df_final.to_pickle(path_matrix)
            print(f"Matrice sauvegardée avec succès sous '{path_matrix}'.")
        else:
            return None, None

    # On sépare X et Y pour l'export
    data_X_formatee = df_final.drop(columns=['quality']).astype(float)
    data_Y_formatee = df_final['quality']
    
    return data_X_formatee, data_Y_formatee

# EXÉCUTION AUTOMATIQUE lors de l'import
data_X_formatee, data_Y_formatee = get_or_create_formatted_data()