import pandas as pd
import numpy as np
import pickle

from scipy.stats import poisson

from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

# =====================================================
# CONFIG
# =====================================================

MAX_GOLES = 6

# =====================================================
# DIXON COLES
# =====================================================

RHO = -0.08

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
# CARGAR DATOS
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
# SPLIT TEMPORAL
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
# CARGAR MODELOS
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
# VARIABLES
# =====================================================

X_test = test_df.drop(
    columns=[
        "date",
        "target_home",
        "target_away"
    ]
)

# =====================================================
# CONTADORES
# =====================================================

aciertos = 0

y_real = []
y_pred = []

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

    pred = np.argmax(
        [
            p_local,
            p_empate,
            p_visitante
        ]
    )

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

    if pred == real:

        aciertos += 1

    y_real.append(
        real
    )

    y_pred.append(
        pred
    )

# =====================================================
# ACCURACY GENERAL
# =====================================================

accuracy = (
    aciertos
    /
    len(test_df)
)

print("\n====================")
print("BACKTESTING")
print("====================")

print(
    f"Accuracy: {accuracy:.4%}"
)

print(
    f"Aciertos: {aciertos}"
)

print(
    f"Partidos: {len(test_df)}"
)

# =====================================================
# MATRIZ DE CONFUSION
# =====================================================

cm = confusion_matrix(
    y_real,
    y_pred
)

print("\n====================")
print("MATRIZ DE CONFUSION")
print("====================")

print(cm)

# =====================================================
# ACCURACY POR CLASE
# =====================================================

print("\n====================")
print("ACCURACY POR CLASE")
print("====================")

clases = [
    "Local",
    "Empate",
    "Visitante"
]

for i in range(3):

    total = cm[i].sum()

    if total > 0:

        acc = (
            cm[i, i]
            /
            total
        )

        print(
            f"{clases[i]}: {acc:.2%}"
        )

# =====================================================
# REPORTE COMPLETO
# =====================================================

print("\n====================")
print("REPORTE")
print("====================")

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
