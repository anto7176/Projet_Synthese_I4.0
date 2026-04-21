import pandas as pd

def load_data(path_x: str, path_y: str):
    try:
        data_x = pd.read_csv(path_x)
        print(f"Succès : {path_x} a été importé avec {len(data_x)} lignes.")
    except FileNotFoundError:
        print(f"Erreur : Le fichier {path_x} est introuvable.")
        data_x = None

    try:
        data_y = pd.read_csv(path_y)
        print(f"Succès : {path_y} a été importé avec {len(data_y)} lignes.")
    except FileNotFoundError:
        print(f"Erreur : Le fichier {path_y} est introuvable.")
        data_y = None
        
    return data_x, data_y

if __name__ == "__main__":
    df_x, df_y = load_data("data/data_X.csv", "data/data_Y.csv")