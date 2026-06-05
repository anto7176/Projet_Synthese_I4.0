import pandas as pd
import numpy as np


def charger_donnees(chemin_x: str, chemin_y: str):
    try:
        donnees_x = pd.read_csv(chemin_x)
        print(f"Import X : {len(donnees_x)} lignes")
    except FileNotFoundError:
        print(f"Erreur : fichier introuvable — {chemin_x}")
        donnees_x = None

    try:
        donnees_y = pd.read_csv(chemin_y)
        print(f"Import Y : {len(donnees_y)} lignes")
    except FileNotFoundError:
        print(f"Erreur : fichier introuvable — {chemin_y}")
        donnees_y = None

    return donnees_x, donnees_y


def nettoyer_donnees(df):
    if df is None:
        return None

    df_propre = df.copy()

    # Si un capteur dévie de plus de 50 par rapport à ses deux voisins de chambre,
    # on le remplace par la moyenne des deux autres (capteur isolé défaillant)
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

    print(f"Nettoyage : {len(df)} → {len(df_propre)} lignes ({len(df) - len(df_propre)} supprimées)")
    return df_propre


def normaliser_qualite(df_y, colonne='quality'):
    if df_y is None:
        return None

    df_norm = df_y.copy()
    q_min = df_norm[colonne].min()
    q_max = df_norm[colonne].max()
    df_norm[colonne] = (df_norm[colonne] - q_min) / (q_max - q_min) * 100

    print(f"Normalisation quality : [{q_min:.2f}, {q_max:.2f}] → [0, 100]")
    return df_norm


def formater_index_temporel(df, nom_colonne):
    if df is None:
        return None

    df[nom_colonne] = pd.to_datetime(df[nom_colonne])
    df = df.set_index(nom_colonne)
    df = df.sort_index()
    return df


if __name__ == "__main__":

    df_x, df_y = charger_donnees("data/data_X.csv", "data/data_Y.csv")

    if df_x is not None and df_y is not None:

        df_x_propre = nettoyer_donnees(df_x)
        df_y_norm = normaliser_qualite(df_y, colonne='quality')

        df_x_final = formater_index_temporel(df_x_propre, "date_time")
        df_y_final = formater_index_temporel(df_y_norm, "date_time")

        print(df_x_final.head(10))
        print(df_x_final.describe().T[['min', 'max']])
        print(df_y_final['quality'].describe())
