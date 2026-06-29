import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

# ==========================================================
# CONFIGURACIÓN GENERAL
# ==========================================================

N_PARTIDOS = 30

DECAY_FACTOR = 0.002

URL = (
    "https://raw.githubusercontent.com/"
    "martj42/international_results/master/results.csv"
)

# ==========================================================
# EQUIPOS DEL MUNDIAL 2026
# ==========================================================

GRUPOS = {

    "A": [
        "Mexico",
        "South Africa",
        "South Korea",
        "Czech Republic"
    ],

    "B": [
        "Canada",
        "Bosnia and Herzegovina",
        "Qatar",
        "Switzerland"
    ],

    "C": [
        "Brazil",
        "Morocco",
        "Haiti",
        "Scotland"
    ],

    "D": [
        "United States",
        "Paraguay",
        "Australia",
        "Turkey"
    ],

    "E": [
        "Germany",
        "Curaçao",
        "Ivory Coast",
        "Ecuador"
    ],

    "F": [
        "Netherlands",
        "Japan",
        "Sweden",
        "Tunisia"
    ],

    "G": [
        "Belgium",
        "Egypt",
        "Iran",
        "New Zealand"
    ],

    "H": [
        "Spain",
        "Cape Verde",
        "Saudi Arabia",
        "Uruguay"
    ],

    "I": [
        "France",
        "Senegal",
        "Iraq",
        "Norway"
    ],

    "J": [
        "Argentina",
        "Algeria",
        "Austria",
        "Jordan"
    ],

    "K": [
        "Portugal",
        "DR Congo",
        "Uzbekistan",
        "Colombia"
    ],

    "L": [
        "England",
        "Croatia",
        "Ghana",
        "Panama"
    ]
}

EQUIPOS_MUNDIAL = []

for grupo in GRUPOS.values():
    EQUIPOS_MUNDIAL.extend(grupo)

# ==========================================================
# DESCARGA AUTOMÁTICA
# ==========================================================

def cargar_datos():

    print("\nDescargando datos...")

    df = pd.read_csv(URL)

    df["date"] = pd.to_datetime(df["date"])

    print("Datos descargados correctamente")

    return df

# ==========================================================
# PESO POR COMPETICIÓN
# ==========================================================

def peso_competicion(torneo):

    torneo = str(torneo).lower()

    if "world cup" in torneo:
        return 1.50

    elif "qualification" in torneo:
        return 1.25

    elif "nations" in torneo:
        return 1.10

    elif "euro" in torneo:
        return 1.30

    elif "copa america" in torneo:
        return 1.30

    elif "gold cup" in torneo:
        return 1.15

    else:
        return 0.75

# ==========================================================
# ÚLTIMOS PARTIDOS
# ==========================================================

def obtener_ultimos_partidos(df, equipo):

    partidos = df[
        (df["home_team"] == equipo)
        |
        (df["away_team"] == equipo)
    ]

    partidos = partidos.sort_values(
        "date",
        ascending=False
    )

    return partidos.head(N_PARTIDOS)

# ==========================================================
# DATASET MUNDIAL
# ==========================================================

def construir_dataset(df):

    lista = []

    print("\nBuscando partidos...")

    for equipo in EQUIPOS_MUNDIAL:

        partidos = obtener_ultimos_partidos(
            df,
            equipo
        )

        print(
            f"{equipo:<20} {len(partidos)} partidos"
        )

        lista.append(partidos)

    dataset = pd.concat(
        lista,
        ignore_index=True
    )

    dataset = dataset.drop_duplicates()

    return dataset

# ==========================================================
# PESOS TEMPORALES
# ==========================================================

def agregar_pesos(dataset):

    fecha_max = dataset["date"].max()

    dataset["dias"] = (
        fecha_max
        - dataset["date"]
    ).dt.days

    dataset["peso_tiempo"] = np.exp(
        -DECAY_FACTOR *
        dataset["dias"]
    )

    dataset["peso_torneo"] = dataset[
        "tournament"
    ].apply(
        peso_competicion
    )

    dataset["peso_total"] = (
        dataset["peso_tiempo"]
        *
        dataset["peso_torneo"]
    )

    return dataset

# ==========================================================
# RESUMEN
# ==========================================================

def resumen(dataset):

    print("\n")
    print("=" * 70)

    print("RESUMEN DEL DATASET")

    print("=" * 70)

    print(
        f"Partidos: {len(dataset)}"
    )

    print(
        f"Equipos: {len(EQUIPOS_MUNDIAL)}"
    )

    print(
        f"Fecha mínima: {dataset['date'].min()}"
    )

    print(
        f"Fecha máxima: {dataset['date'].max()}"
    )

    print("\nPesos")

    print(
        dataset["peso_total"].describe()
    )

# ==========================================================
# GRÁFICA
# ==========================================================

def graficar(dataset):

    plt.figure(
        figsize=(10,5)
    )

    sns.histplot(
        dataset["peso_total"],
        bins=40,
        kde=True
    )

    plt.title(
        "Distribución de pesos finales"
    )

    plt.tight_layout()

    plt.show()

# ==========================================================
# MAIN
# ==========================================================

if __name__ == "__main__":

    df = cargar_datos()

    df = df.dropna(
        subset=[
            "home_score",
            "away_score"
        ]
    )

    df = df.sort_values(
        "date"
    )

    dataset = construir_dataset(df)

    dataset = agregar_pesos(dataset)

    dataset.to_csv(
        "dataset_mundial_2026.csv",
        index=False
    )

    print(
        "\nArchivo guardado:"
    )

    print(
        "dataset_mundial_2026.csv"
    )

    resumen(dataset)

    graficar(dataset)