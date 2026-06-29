import pandas as pd
import numpy as np
import pickle

from scipy.stats import poisson
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

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
            1
            - rho
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
# BUSQUEDA DE UMBRAL
# =====================================================

resultados = []

umbrales = np.arange(
    0.20,
    0.41,
    0.02
)

for umbral in umbrales:

    y_real = []
    y_pred = []

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

        # =============================================
        # REGLA EMPATE
        # =============================================

        if p_empate >= umbral:

            pred = 1

        else:

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

        y_real.append(
            real
        )

        y_pred.append(
            pred
        )

    # =================================================
    # METRICAS
    # =================================================

    accuracy = accuracy_score(
        y_real,
        y_pred
    )

    cm = confusion_matrix(
        y_real,
        y_pred
    )

    if cm.shape == (3, 3):

        recall_empate = (

            cm[1, 1]
            /
            cm[1].sum()

        )

    else:

        recall_empate = 0

    resultados.append({

        "umbral":
            round(
                umbral,
                2
            ),

        "accuracy":
            accuracy,

        "recall_empate":
            recall_empate

    })

# =====================================================
# RESULTADOS
# =====================================================

resultados = pd.DataFrame(
    resultados
)

resultados = resultados.sort_values(
    "accuracy",
    ascending=False
)

print("\n====================================")
print("BUSQUEDA DE UMBRAL")
print("====================================\n")

print(
    resultados.to_string(
        index=False
    )
)

# =====================================================
# MEJOR UMBRAL
# =====================================================

mejor = resultados.iloc[0]

print("\n====================================")
print("MEJOR UMBRAL")
print("====================================")

print(
    f"Umbral: {mejor['umbral']:.2f}"
)

print(
    f"Accuracy: {mejor['accuracy']:.4%}"
)

print(
    f"Recall Empate: {mejor['recall_empate']:.4%}"
)

# =====================================================
# GUARDAR
# =====================================================

resultados.to_csv(
    "busqueda_umbral_empate.csv",
    index=False
)

print("\nArchivo generado:")
print("busqueda_umbral_empate.csv")