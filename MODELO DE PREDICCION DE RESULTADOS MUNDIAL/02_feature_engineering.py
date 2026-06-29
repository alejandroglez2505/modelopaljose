'''
este codigo genera:
ataque
defensa
forma_5
forma_10
forma_30
puntos_ponderados
'''
import pandas as pd
import numpy as np

# =====================================================
# CARGAR DATASET
# =====================================================

print("Cargando dataset...")

df = pd.read_csv("dataset_mundial_2026.csv")

df["date"] = pd.to_datetime(df["date"])

# =====================================================
# EQUIPOS
# =====================================================

EQUIPOS_MUNDIAL = [
    "Mexico","South Africa","South Korea","Czech Republic",
    "Canada","Bosnia and Herzegovina","Qatar","Switzerland",
    "Brazil","Morocco","Haiti","Scotland",
    "United States","Paraguay","Australia","Turkey",
    "Germany","Curaçao","Ivory Coast","Ecuador",
    "Netherlands","Japan","Sweden","Tunisia",
    "Belgium","Egypt","Iran","New Zealand",
    "Spain","Cape Verde","Saudi Arabia","Uruguay",
    "France","Senegal","Iraq","Norway",
    "Argentina","Algeria","Austria","Jordan",
    "Portugal","DR Congo","Uzbekistan","Colombia",
    "England","Croatia","Ghana","Panama"
]

equipos = EQUIPOS_MUNDIAL
print("\nEquipos del modelo:")

for equipo in equipos:
    print(equipo)
    
print(f"Equipos encontrados: {len(equipos)}")

# =====================================================
# OBTENER PARTIDOS DE UN EQUIPO
# =====================================================

def obtener_partidos_equipo(equipo):

    partidos = df[
        (df["home_team"] == equipo) |
        (df["away_team"] == equipo)
    ].copy()

    partidos = partidos.sort_values(
        "date",
        ascending=False
    )

    return partidos

# =====================================================
# CALCULAR FEATURES
# =====================================================

def calcular_features(equipo):

    partidos = obtener_partidos_equipo(equipo)

    goles_favor = []
    goles_contra = []
    puntos = []
    pesos = []

    for _, row in partidos.iterrows():

        if row["home_team"] == equipo:
            gf = row["home_score"]
            gc = row["away_score"]
        else:
            gf = row["away_score"]
            gc = row["home_score"]

        peso = row["peso_total"]

        goles_favor.append(gf)
        goles_contra.append(gc)
        pesos.append(peso)

        if gf > gc:
            puntos.append(3)
        elif gf == gc:
            puntos.append(1)
        else:
            puntos.append(0)

    goles_favor = np.array(goles_favor, dtype=float)
    goles_contra = np.array(goles_contra, dtype=float)
    puntos = np.array(puntos, dtype=float)
    pesos = np.array(pesos, dtype=float)

    # -----------------------------
    # ATAQUE Y DEFENSA PONDERADOS
    # -----------------------------

    ataque = np.average(
        goles_favor,
        weights=pesos
    )

    defensa = np.average(
        goles_contra,
        weights=pesos
    )

    diferencia_goles = ataque - defensa

    # -----------------------------
    # FORMAS
    # -----------------------------

    forma_5 = np.mean(puntos[:5]) / 3

    forma_10 = np.mean(
        puntos[:min(10, len(puntos))]
    ) / 3

    forma_30 = np.mean(puntos) / 3

    # -----------------------------
    # PUNTOS PONDERADOS
    # -----------------------------

    puntos_ponderados = (
        np.average(
            puntos,
            weights=pesos
        ) / 3
    )

    # -----------------------------
    # RESULTADOS
    # -----------------------------

    victorias = np.mean(
        puntos == 3
    )

    empates = np.mean(
        puntos == 1
    )

    derrotas = np.mean(
        puntos == 0
    )

    return {
        "equipo": equipo,

        "ataque": ataque,
        "defensa": defensa,

        "diferencia_goles":
            diferencia_goles,

        "forma_5":
            forma_5,

        "forma_10":
            forma_10,

        "forma_30":
            forma_30,

        "puntos_ponderados":
            puntos_ponderados,

        "victorias":
            victorias,

        "empates":
            empates,

        "derrotas":
            derrotas
    }

# =====================================================
# GENERAR FEATURES
# =====================================================

print("Calculando features...")

features = []

for equipo in equipos:

    features.append(
        calcular_features(
            equipo
        )
    )

features = pd.DataFrame(features)
print(features.columns)

# =====================================================
# BLOQUE 2.2
# ELO DINÁMICO MUNDIALISTA
# =====================================================

print("\nCalculando ELO dinámico...")

elo = {}

for equipo in equipos:
    elo[equipo] = 1500

partidos_ordenados = df.sort_values("date")

for _, row in partidos_ordenados.iterrows():

    local = row["home_team"]
    visita = row["away_team"]

    if (
        local not in equipos
        or
        visita not in equipos
    ):
        continue

    goles_local = row["home_score"]
    goles_visita = row["away_score"]

    peso = row["peso_total"]

    elo_local = elo[local]
    elo_visita = elo[visita]

    esperado_local = (
        1 /
        (
            1 +
            10 ** (
                (elo_visita - elo_local)
                / 400
            )
        )
    )

    esperado_visita = (
        1 - esperado_local
    )

    if goles_local > goles_visita:

        resultado_local = 1
        resultado_visita = 0

    elif goles_local < goles_visita:

        resultado_local = 0
        resultado_visita = 1

    else:

        resultado_local = 0.5
        resultado_visita = 0.5

    diferencia = abs(
        goles_local -
        goles_visita
    )

    multiplicador = (
        1 +
        diferencia * 0.10
    )

    K = 10 * np.sqrt(peso) * multiplicador

    elo[local] = (
        elo_local +
        K *
        (
            resultado_local -
            esperado_local
        )
    )

    elo[visita] = (
        elo_visita +
        K *
        (
            resultado_visita -
            esperado_visita
        )
    )
    
# =====================================================
# AGREGAR ELO AL DATAFRAME
# =====================================================

features["elo"] = (
    features["equipo"]
    .map(elo)
)

features = features.sort_values(
    "elo",
    ascending=False
)

print("\nTOP 20 ELO\n")

print(
    features[
        [
            "equipo",
            "elo",
            "ataque",
            "defensa"
        ]
    ]
    .head(20)
    .to_string(index=False)
)
# =====================================================
# ORDENAR
# =====================================================

features = features.sort_values(
    "puntos_ponderados",
    ascending=False
)

# =====================================================
# GUARDAR
# =====================================================

features.to_csv(
    "features_equipos.csv",
    index=False
)

# =====================================================
# MOSTRAR RESULTADOS
# =====================================================

print("\nTOP 20 EQUIPOS\n")

print(
    features[
        [
            "equipo",
            "ataque",
            "defensa",
            "forma_30",
            "puntos_ponderados"
        ]
    ].head(20)
)

print("\nArchivo generado:")
print("features_equipos.csv")

# =====================================================
# BLOQUE 2.3
# RATING ML COMPUESTO
# =====================================================

print("\nCalculando Rating ML...")

# -------------------------
# NORMALIZACIÓN
# -------------------------

features["elo_norm"] = (
    features["elo"] - features["elo"].min()
) / (
    features["elo"].max() - features["elo"].min()
)

features["ataque_norm"] = (
    features["ataque"] - features["ataque"].min()
) / (
    features["ataque"].max() - features["ataque"].min()
)

# Menos goles recibidos = mejor defensa
features["defensa_norm"] = 1 - (
    (features["defensa"] - features["defensa"].min())
    /
    (features["defensa"].max() - features["defensa"].min())
)

# -------------------------
# RATING COMPUESTO
# -------------------------

features["rating_ml"] = (
    0.40 * features["elo_norm"]
    +
    0.30 * features["ataque_norm"]
    +
    0.20 * features["defensa_norm"]
    +
    0.10 * features["forma_30"]
)

# Escala 0-100
features["rating_ml"] = (
    features["rating_ml"] * 100
)

# =====================================================
# RANKING
# =====================================================

features = features.sort_values(
    "rating_ml",
    ascending=False
)

print("\n")
print("=" * 80)
print("TOP 20 RATING ML")
print("=" * 80)

print(
    features[
        [
            "equipo",
            "rating_ml",
            "elo",
            "ataque",
            "defensa",
            "forma_30"
        ]
    ]
    .head(20)
    .to_string(index=False)
)
# =====================================================
# BLOQUE 2.4
# FUERZA DEL RIVAL
# =====================================================

print("\nCalculando fuerza de rival...")

rating_dict = dict(
    zip(
        features["equipo"],
        features["rating_ml"]
    )
)

fuerza_rival = []

for equipo in features["equipo"]:

    partidos = obtener_partidos_equipo(
        equipo
    )

    ratings_rivales = []
    pesos_rivales = []

    for _, row in partidos.iterrows():

        if row["home_team"] == equipo:
            rival = row["away_team"]
        else:
            rival = row["home_team"]

        if rival not in rating_dict:
            continue

        ratings_rivales.append(
            rating_dict[rival]
        )

        pesos_rivales.append(
            row["peso_total"]
        )

    if len(ratings_rivales) > 0:

        fuerza = np.average(
            ratings_rivales,
            weights=pesos_rivales
        )

    else:

        fuerza = np.nan

    fuerza_rival.append(
        fuerza
    )

features["fuerza_rival"] = fuerza_rival

# =====================================================
# BLOQUE 2.5
# ATAQUE AJUSTADO
# =====================================================

promedio_fuerza = (
    features["fuerza_rival"]
    .mean()
)

features["ataque_ajustado"] = (

    features["ataque"]

    *

    (
        features["fuerza_rival"]
        /
        promedio_fuerza
    )
)

# =====================================================
# BLOQUE 2.6
# DEFENSA AJUSTADA
# =====================================================

features["defensa_ajustada"] = (

    features["defensa"]

    /

    (
        features["fuerza_rival"]
        /
        promedio_fuerza
    )
)

# =====================================================
# RATING FINAL
# =====================================================

features["rating_final"] = (

    0.35 *

    (
        features["rating_ml"]
        /
        features["rating_ml"].max()
    )

    +

    0.35 *

    (
        features["ataque_ajustado"]
        /
        features["ataque_ajustado"].max()
    )

    +

    0.30 *

    (
        1
        -
        (
            features["defensa_ajustada"]
            /
            features["defensa_ajustada"].max()
        )
    )

)

features["rating_final"] = (
    features["rating_final"]
    * 100
)

# =====================================================
# ORDEN FINAL
# =====================================================

features = features.sort_values(
    "rating_final",
    ascending=False
)

# =====================================================
# MOSTRAR RESULTADOS
# =====================================================

print("\n")
print("=" * 90)
print("TOP 20 RATING FINAL")
print("=" * 90)

print(
    features[
        [
            "equipo",
            "rating_final",
            "rating_ml",
            "elo",
            "fuerza_rival",
            "ataque_ajustado",
            "defensa_ajustada"
        ]
    ]
    .head(20)
    .to_string(index=False)
)

print("\n")
print("=" * 90)
print("TOP 10 ATAQUES AJUSTADOS")
print("=" * 90)

print(
    features[
        [
            "equipo",
            "ataque_ajustado"
        ]
    ]
    .sort_values(
        "ataque_ajustado",
        ascending=False
    )
    .head(10)
    .to_string(index=False)
)

print("\n")
print("=" * 90)
print("TOP 10 DEFENSAS AJUSTADAS")
print("=" * 90)

print(
    features[
        [
            "equipo",
            "defensa_ajustada"
        ]
    ]
    .sort_values(
        "defensa_ajustada"
    )
    .head(10)
    .to_string(index=False)
)