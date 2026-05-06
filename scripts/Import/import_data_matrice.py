import os
import sys
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from import_data import charger_donnees, nettoyer_donnees, normaliser_qualite, formater_index_temporel


def creer_process_matrix(df_x_final, df_y_final):
    print("Construction de la process matrix (pivot)...")
    
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