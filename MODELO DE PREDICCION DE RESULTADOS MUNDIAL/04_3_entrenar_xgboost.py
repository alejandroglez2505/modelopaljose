import pandas as pd

import numpy as np
import pickle

from xgboost import XGBRegressor

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error

# =====================================================
# CARGAR DATASET
# =====================================================

print("Cargando dataset...")

df = pd.read_csv(
    "dataset_xgboost_final.csv"
)

# =====================================================
# VERIFICAR FECHA
# =====================================================

if "date" not in df.columns:

    raise Exception(
        "No existe la columna 'date'. "
        "Vuelve a ejecutar:\n"
        "04_1_crear_dataset_xgb.py\n"
        "04_2_agregar_pymc.py"
    )

df["date"] = pd.to_datetime(
    df["date"]
)

# =====================================================
# ORDEN CRONOLÓGICO
# =====================================================

df = df.sort_values(
    "date"
).reset_index(drop=True)

print(
    f"Partidos totales: {len(df)}"
)

# =====================================================
# SPLIT TEMPORAL
# =====================================================

corte = int(
    len(df) * 0.80
)

train_df = df.iloc[:corte].copy()
test_df = df.iloc[corte:].copy()

print(
    f"Partidos entrenamiento: {len(train_df)}"
)

print(
    f"Partidos prueba: {len(test_df)}"
)

# =====================================================
# VARIABLES X
# =====================================================

columnas_excluir = [

    "date",

    "target_home",
    "target_away"

]

X_train = train_df.drop(
    columns=columnas_excluir
)

X_test = test_df.drop(
    columns=columnas_excluir
)

# =====================================================
# TARGET HOME
# =====================================================

y_home_train = train_df[
    "target_home"
]

y_home_test = test_df[
    "target_home"
]

# =====================================================
# TARGET AWAY
# =====================================================

y_away_train = train_df[
    "target_away"
]

y_away_test = test_df[
    "target_away"
]

# =====================================================
# MODELO LOCAL
# =====================================================

print(
    "\nEntrenando modelo goles local..."
)

modelo_home = XGBRegressor(

    n_estimators=500,

    max_depth=4,

    learning_rate=0.03,

    subsample=0.8,

    colsample_bytree=0.8,

    objective="reg:squarederror",

    random_state=42

)

modelo_home.fit(
    X_train,
    y_home_train
)

# =====================================================
# MODELO VISITANTE
# =====================================================

print(
    "\nEntrenando modelo goles visitante..."
)

modelo_away = XGBRegressor(

    n_estimators=500,

    max_depth=4,

    learning_rate=0.03,

    subsample=0.8,

    colsample_bytree=0.8,

    objective="reg:squarederror",

    random_state=42

)

modelo_away.fit(
    X_train,
    y_away_train
)

# =====================================================
# PREDICCIONES
# =====================================================

pred_home = modelo_home.predict(
    X_test
)

pred_away = modelo_away.predict(
    X_test
)

# =====================================================
# MÉTRICAS
# =====================================================

mae_home = mean_absolute_error(
    y_home_test,
    pred_home
)

mae_away = mean_absolute_error(
    y_away_test,
    pred_away
)

rmse_home = np.sqrt(
    mean_squared_error(
        y_home_test,
        pred_home
    )
)

rmse_away = np.sqrt(
    mean_squared_error(
        y_away_test,
        pred_away
    )
)

# =====================================================
# RESULTADOS
# =====================================================

print("\n==============================")
print("RESULTADOS XGBOOST TEMPORAL")
print("==============================")

print(
    f"MAE Local: {mae_home:.4f}"
)

print(
    f"MAE Visitante: {mae_away:.4f}"
)

print(
    f"RMSE Local: {rmse_home:.4f}"
)

print(
    f"RMSE Visitante: {rmse_away:.4f}"
)

# =====================================================
# IMPORTANCIA VARIABLES
# =====================================================

importancias = pd.DataFrame({

    "variable":
        X_train.columns,

    "importancia":
        modelo_home.feature_importances_

})

importancias = importancias.sort_values(
    by="importancia",
    ascending=False
)

print("\nTOP 20 VARIABLES\n")

print(
    importancias
    .head(20)
    .to_string(index=False)
)

# =====================================================
# GUARDAR MODELOS
# =====================================================

with open(
    "modelo_home_xgb.pkl",
    "wb"
) as f:

    pickle.dump(
        modelo_home,
        f
    )

with open(
    "modelo_away_xgb.pkl",
    "wb"
) as f:

    pickle.dump(
        modelo_away,
        f
    )

print("\nArchivos generados:")

print(
    "modelo_home_xgb.pkl"
)

print(
    "modelo_away_xgb.pkl"
)