import pandas as pd
import numpy as np
import pickle

from scipy.stats import poisson

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score
)

# =====================================================
# CONFIGURACION
# =====================================================

MAX_GOLES = 6

UMBRAL_EMPATE = 0.30

RHO = -0.08

# =====================================================
# DIXON COLES
# =====================================================

def tau_dc(
    x,
    y,
    lambda_home,
    lambda_away,
    rho
):

    if x == 0 and y == 0:

        return (
            1
            - lambda_home
            * lambda_away
            * rho
        )

    elif x == 0 and y == 1:

        return (
            1
            + lambda_home
            * rho
        )

    elif x == 1 and y == 0:

        return (
            1
            + lambda_away
            * rho
        )

    elif x == 1 and y == 1:

        return (
            1 - rho
        )

    return 1


# =====================================================
# CARGAR DATASET
# =====================================================

print("Cargando datos...")

df = pd.read_csv(
    "dataset_xgboost_final.csv"
)

df["date"] = pd.to_datetime(
    df["date"]
)

df = df.sort_values(
    "date"
).reset_index(
    drop=True
)

# =====================================================
# TEST SET
# =====================================================

corte = int(
    len(df) * 0.80
)

test_df = df.iloc[
    corte:
].copy()

print(
    f"Partidos prueba: {len(test_df)}"
)

# =====================================================
# MODELOS
# =====================================================

with open(
    "modelo_home_xgb.pkl",
    "rb"
) as f:

    modelo_home = pickle.load(f)

with open(
    "modelo_away_xgb.pkl",
    "rb"
) as f:

    modelo_away = pickle.load(f)

# =====================================================
# FEATURES
# =====================================================

X_test = test_df.drop(
    columns=[
        "date",
        "target_home",
        "target_away"
    ]
)

# =====================================================
# RESULTADOS
# =====================================================

y_real = []
y_pred = []

predicciones = []

# =====================================================
# LOOP
# =====================================================

for idx in range(
    len(test_df)
):

    fila = X_test.iloc[
        [idx]
    ]

    lambda_home = max(
        0.05,
        modelo_home.predict(
            fila
        )[0]
    )

    lambda_away = max(
        0.05,
        modelo_away.predict(
            fila
        )[0]
    )

    p_local = 0
    p_empate = 0
    p_visitante = 0

    for i in range(
        MAX_GOLES + 1
    ):

        for j in range(
            MAX_GOLES + 1
        ):

            prob = (

                poisson.pmf(
                    i,
                    lambda_home
                )

                *

                poisson.pmf(
                    j,
                    lambda_away
                )

                *

                tau_dc(
                    i,
                    j,
                    lambda_home,
                    lambda_away,
                    RHO
                )

            )

            if i > j:

                p_local += prob

            elif i == j:

                p_empate += prob

            else:

                p_visitante += prob

    # =================================================
    # PREDICCION
    # =================================================

    if p_empate >= UMBRAL_EMPATE:

        pred = 1

    else:

        pred = np.argmax(
            [
                p_local,
                p_empate,
                p_visitante
            ]
        )

    # =================================================
    # RESULTADO REAL
    # =================================================

    real_home = test_df.iloc[idx][
        "target_home"
    ]

    real_away = test_df.iloc[idx][
        "target_away"
    ]

    if real_home > real_away:

        real = 0

    elif real_home == real_away:

        real = 1

    else:

        real = 2

    y_real.append(
        real
    )

    y_pred.append(
        pred
    )

    correcto = (
        pred == real
    )

    predicciones.append({

        "fecha":
            test_df.iloc[idx]["date"],

        "goles_local":
            real_home,

        "goles_visitante":
            real_away,

        "lambda_local":
            round(
                lambda_home,
                3
            ),

        "lambda_visitante":
            round(
                lambda_away,
                3
            ),

        "p_local":
            round(
                p_local,
                4
            ),

        "p_empate":
            round(
                p_empate,
                4
            ),

        "p_visitante":
            round(
                p_visitante,
                4
            ),

        "real":
            real,

        "pred":
            pred,

        "correcto":
            correcto

    })

# =====================================================
# METRICAS
# =====================================================

accuracy = accuracy_score(
    y_real,
    y_pred
)

cm = confusion_matrix(
    y_real,
    y_pred
)

# =====================================================
# RESULTADOS
# =====================================================

print("\n================================")
print("VALIDACION FINAL")
print("================================")

print(
    f"Accuracy: {accuracy:.4%}"
)

print("\n================================")
print("MATRIZ DE CONFUSION")
print("================================")

print(cm)

print("\n================================")
print("REPORTE")
print("================================")

print(

    classification_report(

        y_real,

        y_pred,

        target_names=[

            "Local",

            "Empate",

            "Visitante"

        ]

    )

)

# =====================================================
# GUARDAR CSV
# =====================================================

predicciones = pd.DataFrame(
    predicciones
)

predicciones.to_csv(
    "predicciones_finales.csv",
    index=False
)

print("\nArchivo generado:")
print("predicciones_finales.csv")