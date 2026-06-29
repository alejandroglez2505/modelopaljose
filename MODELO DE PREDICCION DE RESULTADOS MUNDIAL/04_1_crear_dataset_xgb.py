import pandas as pd

print("Cargando archivos...")

# =====================================================
# DATASET DE PARTIDOS
# =====================================================

partidos = pd.read_csv(
    "dataset_mundial_2026.csv"
)

partidos["date"] = pd.to_datetime(
    partidos["date"]
)

# =====================================================
# FEATURES
# =====================================================

features = pd.read_csv(
    "features_equipos.csv"
)

# =====================================================
# HOME FEATURES
# =====================================================

home = features.copy()

home.columns = [

    "home_" + c
    if c != "equipo"
    else "home_team"

    for c in home.columns

]

# =====================================================
# AWAY FEATURES
# =====================================================

away = features.copy()

away.columns = [

    "away_" + c
    if c != "equipo"
    else "away_team"

    for c in away.columns

]

# =====================================================
# MERGE HOME
# =====================================================

df = partidos.merge(

    home,

    on="home_team",

    how="left"

)

# =====================================================
# MERGE AWAY
# =====================================================

df = df.merge(

    away,

    on="away_team",

    how="left"

)

# =====================================================
# TARGETS
# =====================================================

df["target_home"] = df["home_score"]

df["target_away"] = df["away_score"]

# =====================================================
# ELIMINAR SOLO LO INNECESARIO
# =====================================================

columnas_borrar = [

    "tournament",
    "city",
    "country",

    "home_score",
    "away_score",

    "home_team",
    "away_team"

]

for columna in columnas_borrar:

    if columna in df.columns:

        df.drop(
            columns=[columna],
            inplace=True
        )

# =====================================================
# ORDEN TEMPORAL
# =====================================================

df = df.sort_values(
    "date"
).reset_index(
    drop=True
)

# =====================================================
# GUARDAR
# =====================================================

df.to_csv(

    "dataset_xgboost.csv",

    index=False

)

print("\nArchivo generado")

print(
    "dataset_xgboost.csv"
)

print(
    f"\nFilas: {len(df)}"
)

print(
    f"Columnas: {len(df.columns)}"
)

print(
    "\nColumnas:"
)

print(
    df.columns.tolist()
)

print(
    "\nFecha mínima:",
    df["date"].min()
)

print(
    "Fecha máxima:",
    df["date"].max()
)