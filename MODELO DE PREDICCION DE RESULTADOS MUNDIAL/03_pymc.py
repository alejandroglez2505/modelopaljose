import pandas as pd
import numpy as np
import pymc as pm
import pickle


def main():

    # =====================================================
    # CARGAR DATASET
    # =====================================================

    print("Cargando datos...")

    df = pd.read_csv(
        "dataset_mundial_2026.csv"
    )

    # =====================================================
    # MUNDIALISTAS 2026
    # =====================================================

    EQUIPOS_MUNDIAL = [
        "Mexico", "South Africa", "South Korea", "Czech Republic",
        "Canada", "Bosnia and Herzegovina", "Qatar", "Switzerland",
        "Brazil", "Morocco", "Haiti", "Scotland",
        "United States", "Paraguay", "Australia", "Turkey",
        "Germany", "Curaçao", "Ivory Coast", "Ecuador",
        "Netherlands", "Japan", "Sweden", "Tunisia",
        "Belgium", "Egypt", "Iran", "New Zealand",
        "Spain", "Cape Verde", "Saudi Arabia", "Uruguay",
        "France", "Senegal", "Iraq", "Norway",
        "Argentina", "Algeria", "Austria", "Jordan",
        "Portugal", "DR Congo", "Uzbekistan", "Colombia",
        "England", "Croatia", "Ghana", "Panama"
    ]

    # =====================================================
    # FILTRAR SOLO PARTIDOS ENTRE MUNDIALISTAS
    # =====================================================

    df = df[
        (df["home_team"].isin(EQUIPOS_MUNDIAL))
        &
        (df["away_team"].isin(EQUIPOS_MUNDIAL))
    ].copy()

    print(
        f"Partidos utilizados: {len(df)}"
    )

    # =====================================================
    # EQUIPOS
    # =====================================================

    equipos = sorted(
        list(
            set(df["home_team"])
            |
            set(df["away_team"])
        )
    )

    n_teams = len(equipos)

    print(
        f"Equipos encontrados: {n_teams}"
    )

    equipo_idx = {
        equipo: i
        for i, equipo in enumerate(equipos)
    }

    # =====================================================
    # ÍNDICES
    # =====================================================

    home_idx = (
        df["home_team"]
        .map(equipo_idx)
        .values
    )

    away_idx = (
        df["away_team"]
        .map(equipo_idx)
        .values
    )

    home_goals = (
        df["home_score"]
        .values
    )

    away_goals = (
        df["away_score"]
        .values
    )

    # =====================================================
    # MODELO BAYESIANO
    # =====================================================

    print(
        "\nEntrenando PyMC..."
    )

    with pm.Model() as modelo:

        sigma_att = pm.HalfNormal(
            "sigma_att",
            sigma=1
        )

        sigma_def = pm.HalfNormal(
            "sigma_def",
            sigma=1
        )

        # -----------------------------------------
        # ATAQUE CENTRADO
        # -----------------------------------------

        ataque_raw = pm.Normal(
            "ataque_raw",
            mu=0,
            sigma=sigma_att,
            shape=n_teams
        )

        ataque = pm.Deterministic(
            "ataque",
            ataque_raw - pm.math.mean(ataque_raw)
        )

        # -----------------------------------------
        # DEFENSA CENTRADA
        # -----------------------------------------

        defensa_raw = pm.Normal(
            "defensa_raw",
            mu=0,
            sigma=sigma_def,
            shape=n_teams
        )

        defensa = pm.Deterministic(
            "defensa",
            defensa_raw - pm.math.mean(defensa_raw)
        )

        intercept = pm.Normal(
            "intercept",
            mu=0,
            sigma=1
        )

        home_adv = pm.Normal(
            "home_adv",
            mu=0,
            sigma=1
        )

        lambda_home = pm.math.exp(
            intercept
            + ataque[home_idx]
            - defensa[away_idx]
            + home_adv
        )

        lambda_away = pm.math.exp(
            intercept
            + ataque[away_idx]
            - defensa[home_idx]
        )

        pm.Poisson(
            "home_score",
            mu=lambda_home,
            observed=home_goals
        )

        pm.Poisson(
            "away_score",
            mu=lambda_away,
            observed=away_goals
        )

        trace = pm.sample(
            draws=1000,
            tune=1000,
            chains=1,
            cores=1,
            target_accept=0.90,
            progressbar=True,
            return_inferencedata=True
        )

    # =====================================================
    # GUARDAR TRACE
    # =====================================================

    with open(
        "trace.pkl",
        "wb"
    ) as f:

        pickle.dump(
            trace,
            f
        )

    # =====================================================
    # EXTRAER RATINGS
    # =====================================================

    ataque_media = (
        trace.posterior["ataque"]
        .mean(
            dim=("chain", "draw")
        )
        .values
    )

    defensa_media = (
        trace.posterior["defensa"]
        .mean(
            dim=("chain", "draw")
        )
        .values
    )

    ratings = pd.DataFrame({
        "equipo": equipos,
        "ataque_pymc": ataque_media,
        "defensa_pymc": defensa_media
    })

    ratings = ratings.sort_values(
        "ataque_pymc",
        ascending=False
    )

    ratings.to_csv(
        "ratings_pymc.csv",
        index=False
    )

    # =====================================================
    # RESULTADOS
    # =====================================================

    print("\nTOP 20 ATAQUES PYMC\n")

    print(
        ratings.head(20)
        .to_string(index=False)
    )

    print("\nArchivo generado:")

    print("ratings_pymc.csv")
    print("trace.pkl")


if __name__ == "__main__":
    main()