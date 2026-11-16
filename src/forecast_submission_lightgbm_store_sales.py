"""Prévision LightGBM (modèle global) pour la compétition Kaggle
*Store Sales - Time Series Forecasting* (54 magasins x 33 familles, 1782
entités) — benchmark indépendant de l'app, pour comparer contre SARIMAX
(0.42726 RMSLE, meilleur score actuel, cf. README) et juger si les pistes
identifiées (lag annuel, réglage hyperparamètres/CV, feature de volume
historique) suffisent à le dépasser.

Volontairement séparé de src/lightgbm_forecast.py (utilisé par l'app) :
dataset, features (promotions, pétrole, jours fériés Équateur, magasin) et
hyperparamètres n'ont rien à voir avec le cas d'usage générique de l'app.

Deux modes :
- sans argument (défaut) : backtest local sur les 16 derniers jours connus de
  train.csv (même horizon que le test Kaggle réel) — calcule un RMSLE
  comparable, sans consommer de soumission Kaggle.
- `--submit` : entraîne sur tout l'historique et écrit le CSV à uploader sur
  Kaggle (horizon officiel 2017-08-16 -> 2017-08-31).

`--family "<NOM>"` restreint les deux modes à une seule famille, pour tester
un modèle dédié plutôt que le modèle global sur les 33 familles.
"""

import argparse
import os
import time

import lightgbm as lgb
import numpy as np
import pandas as pd

DATA_DIR = os.environ.get(
    "STORE_SALES_DIR", os.path.expanduser("~/Téléchargements/store-sales-time-series-forecasting"))

HORIZON = 16
LAG_DAYS = (1, 2, 3, 4, 5, 6, 7, 14, 21, 28, 364)
ROLLING_WINDOWS = (7, 14, 28, 84)
TAIL_WINDOW_DAYS = 400  # >= max(lag, rolling) + marge, pour la boucle récursive

CATEGORICAL_COLS = ["store_nbr__code", "family__code", "city__code", "state__code", "type__code", "cluster"]

LGB_PARAMS = {
    # tweedie plutôt que L2 sur log1p(ventes) : la cible est très asymétrique
    # et zero-inflated (beaucoup d'entités à ventes nulles ou quasi nulles,
    # quelques-unes avec des pics x10-x20, ex. rentrée scolaire) — un modèle
    # global entraîné en L2 régresse ces pics vers la moyenne, dilués dans
    # 3M lignes à faible volume. Tweedie (variance_power proche de 1 =
    # proche Poisson) est l'objectif standard pour ce type de cible sur ce
    # jeu de données précis.
    "objective": "tweedie",
    "tweedie_variance_power": 1.1,
    "metric": "rmse",
    "learning_rate": 0.03,
    "num_leaves": 255,
    "min_data_in_leaf": 10,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "lambda_l2": 0.05,
    "verbosity": -1,
    "num_threads": os.cpu_count() or 4,
}
NUM_BOOST_ROUND = 2000
EARLY_STOPPING_ROUNDS = 100
# figé après tuning par --validate : nombre
# d'itérations retenu pour l'entraînement final sans validation interne.
FINAL_NUM_BOOST_ROUND = 1200


def load_raw() -> dict[str, pd.DataFrame]:
    train = pd.read_csv(f"{DATA_DIR}/train.csv", parse_dates=["date"])
    test = pd.read_csv(f"{DATA_DIR}/test.csv", parse_dates=["date"])
    stores = pd.read_csv(f"{DATA_DIR}/stores.csv")
    oil = pd.read_csv(f"{DATA_DIR}/oil.csv", parse_dates=["date"])
    holidays = pd.read_csv(f"{DATA_DIR}/holidays_events.csv", parse_dates=["date"])
    transactions = pd.read_csv(f"{DATA_DIR}/transactions.csv", parse_dates=["date"])
    return {
        "train": train, "test": test, "stores": stores, "oil": oil,
        "holidays": holidays, "transactions": transactions,
    }


def build_oil_series(oil: pd.DataFrame, dates: pd.DatetimeIndex) -> pd.Series:
    """Prix quotidien, journées sans cotation (week-ends, fériés US) comblées
    par report de la dernière valeur connue."""
    s = oil.set_index("date")["dcoilwtico"].reindex(dates).ffill().bfill()
    s.index.name = "date"
    return s


def build_holiday_flags(holidays: pd.DataFrame, stores: pd.DataFrame, all_dates: pd.DatetimeIndex) -> pd.DataFrame:
    """Un flag 'is_holiday' par (date, store_nbr), vectorisé (pas de boucle
    par entité) : jour férié national (tous magasins), régional (état du
    magasin) ou local (ville du magasin). 'transferred=True' annule le jour
    d'origine ; le type 'Transfer' porte le jour réellement chômé. Bridge et
    Additional comptent comme fériés. 'Work Day' (compense un pont) et
    'Event' (ex. séisme, Coupe du monde) volontairement ignorés en v1 —
    nuance mineure, hors des pistes identifiées dans le journal.
    """
    effective = holidays[
        ((holidays["type"] == "Holiday") & (~holidays["transferred"]))
        | holidays["type"].isin(["Transfer", "Additional", "Bridge"])
    ]

    national_dates = set(effective.loc[effective["locale"] == "National", "date"])
    regional = effective.loc[effective["locale"] == "Regional", ["date", "locale_name"]].rename(
        columns={"locale_name": "state"}
    )
    local = effective.loc[effective["locale"] == "Local", ["date", "locale_name"]].rename(
        columns={"locale_name": "city"}
    )

    base = pd.MultiIndex.from_product(
        [all_dates, stores["store_nbr"]], names=["date", "store_nbr"]
    ).to_frame(index=False)
    base = base.merge(stores[["store_nbr", "city", "state"]], on="store_nbr", how="left")

    base["is_national_holiday"] = base["date"].isin(national_dates)
    base = base.merge(regional.assign(is_regional_holiday=True), on=["date", "state"], how="left")
    base = base.merge(local.assign(is_local_holiday=True), on=["date", "city"], how="left")
    base["is_holiday"] = (
        base["is_national_holiday"]
        | base["is_regional_holiday"].fillna(False)
        | base["is_local_holiday"].fillna(False)
    ).astype("int8")

    return base[["date", "store_nbr", "is_holiday"]]


def build_transactions_features(
    transactions: pd.DataFrame, all_dates: pd.DatetimeIndex, stores: pd.DataFrame, cutoff: pd.Timestamp
) -> pd.DataFrame:
    """Fréquentation quotidienne par magasin (toutes familles confondues) — signal
    absent de train.csv, qui décompose déjà les ventes par famille. Contrairement à
    oil.csv, transactions.csv ne couvre jamais l'horizon à prévoir (aucune valeur
    après `cutoff`, que ce soit le vrai dernier jour connu en soumission ou un jour
    plus tôt en backtest) : tronqué à `cutoff` avant tout calcul, pour ne jamais
    laisser un vrai backtest voir des transactions "futures" par rapport à lui.

    Décalé d'un jour (causal) sur l'historique connu ; au-delà de `cutoff`, gelé sur
    l'état "à cutoff inclus" plutôt que laissé se diluer en NaN au fil de l'horizon.
    """
    known = transactions[transactions["date"] <= cutoff]
    idx = pd.MultiIndex.from_product([stores["store_nbr"], all_dates], names=["store_nbr", "date"])
    base = idx.to_frame(index=False).merge(known, on=["store_nbr", "date"], how="left")
    base = base.sort_values(["store_nbr", "date"]).reset_index(drop=True)

    g = base.groupby("store_nbr", observed=True)["transactions"]
    base["tx_lag_1"] = g.shift(1)
    base["tx_roll_mean_7"] = g.transform(lambda s: s.shift(1).rolling(7, min_periods=1).mean())
    base["tx_roll_mean_28"] = g.transform(lambda s: s.shift(1).rolling(28, min_periods=1).mean())

    snap_mask = base["date"] <= cutoff
    snapshot = base.loc[snap_mask].groupby("store_nbr", observed=True)["transactions"].agg(
        _tx_last="last",
        _tx_roll7=lambda s: s.tail(7).mean(),
        _tx_roll28=lambda s: s.tail(28).mean(),
    )
    future_mask = ~snap_mask
    base.loc[future_mask, "tx_lag_1"] = base.loc[future_mask, "store_nbr"].map(snapshot["_tx_last"])
    base.loc[future_mask, "tx_roll_mean_7"] = base.loc[future_mask, "store_nbr"].map(snapshot["_tx_roll7"])
    base.loc[future_mask, "tx_roll_mean_28"] = base.loc[future_mask, "store_nbr"].map(snapshot["_tx_roll28"])

    return base[["store_nbr", "date", "tx_lag_1", "tx_roll_mean_7", "tx_roll_mean_28"]]


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    df["dow"] = df["date"].dt.dayofweek
    df["month"] = df["date"].dt.month
    df["day"] = df["date"].dt.day
    df["is_weekend"] = (df["dow"] >= 5).astype("int8")
    days_in_month = df["date"].dt.days_in_month
    df["is_payday"] = ((df["day"] == 15) | (df["day"] == days_in_month)).astype("int8")
    df["year"] = df["date"].dt.year
    return df


def add_lag_features(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    df = df.sort_values([*group_cols, "date"]).reset_index(drop=True)
    df["y"] = np.log1p(df["sales"].clip(lower=0))

    grouped = df.groupby(group_cols, observed=True)["y"]
    for k in LAG_DAYS:
        df[f"lag_{k}"] = grouped.shift(k)
    for w in ROLLING_WINDOWS:
        df[f"roll_mean_{w}"] = grouped.transform(lambda s, w=w: s.shift(1).rolling(w, min_periods=1).mean())

    # tendance d'une année sur l'autre : certaines familles ont une saisonnalité
    # annuelle qui grossit fortement d'une année à l'autre (ex. rentrée scolaire,
    # +73% de 2016 à 2017 sur SCHOOL AND OFFICE SUPPLIES) — un lag_364 brut sous-
    # estime alors largement le pic suivant. roll_mean_28_yoy capture le niveau
    # à la même période l'an dernier (fenêtre, pas un jour isolé, pour lisser le
    # bruit du calendrier) ; yoy_trend, l'écart avec le niveau récent, sert de
    # facteur de croissance que le modèle peut réappliquer.
    df["roll_mean_28_yoy"] = grouped.transform(lambda s: s.shift(1 + 364).rolling(28, min_periods=1).mean())
    df["yoy_trend"] = df["roll_mean_28"] - df["roll_mean_28_yoy"]

    # dynamique récente (accélération) : écart lag_1 vs lag_7, utile pour
    # détecter une montée en cours (ex. rentrée scolaire) indépendamment du
    # niveau absolu, que le lag/roll seuls ne rendent pas explicite.
    df["momentum_1_7"] = df["lag_1"] - df["lag_7"]

    # extrapolation saisonnière x tendance, calculée explicitement en échelle
    # ventes (pas juste laissée à déduire par l'arbre) : un arbre de décision
    # ne peut pas extrapoler au-delà des valeurs vues en feuille (moyenne des
    # exemples d'entraînement), donc une croissance d'une ampleur inédite
    # (ex. rentrée scolaire 2017, +73% vs 2016) reste sous-estimée même avec
    # yoy_trend en feature — testé, gain marginal. Ici, "niveau à la même
    # période l'an dernier (lissé) x ratio de croissance récente" est calculé
    # directement, pour que le modèle n'ait qu'à lire cette valeur plutôt que
    # la reconstruire depuis des features séparées.
    lag_364_smooth = grouped.transform(lambda s: s.shift(361).rolling(7, min_periods=1).mean())
    level_last_year = np.expm1(lag_364_smooth)
    level_recent = np.expm1(df["roll_mean_7"])
    level_recent_last_year = np.expm1(df["roll_mean_28_yoy"])
    # non bornée, le ratio est très bruité sur les entités à faible volume
    # (dénominateur proche de 0) — le rendait peu fiable aux yeux du modèle,
    # y compris là où le signal est net (bornes larges mais explicites).
    growth_ratio = ((level_recent + 1) / (level_recent_last_year + 1)).clip(0.2, 5)
    df["yoy_extrapolation"] = np.log1p(level_last_year * growth_ratio)

    promo_grouped = df.groupby(group_cols, observed=True)["onpromotion"]
    df["roll_mean_promo_7"] = promo_grouped.transform(lambda s: s.shift(1).rolling(7, min_periods=1).mean())

    # signal régional : moyenne (log1p) des ventes du jour pour la même famille,
    # agrégée sur tous les magasins du même cluster (pas seulement ce magasin) —
    # décalé d'un jour (causal, même principe que roll_mean_*). Utile surtout aux
    # paires store x famille à faible volume/historique bruité, qui bénéficient
    # d'un signal plus stable que leur propre historique local.
    cluster_family_daily = (
        df.groupby(["cluster", "family", "date"], observed=True)["y"].mean().reset_index()
        .sort_values(["cluster", "family", "date"])
    )
    cf_grouped = cluster_family_daily.groupby(["cluster", "family"], observed=True)["y"]
    cluster_family_daily["cluster_family_roll_mean_28"] = cf_grouped.transform(
        lambda s: s.shift(1).rolling(28, min_periods=1).mean()
    )
    # `df` peut déjà porter cette colonne (boucle récursive de predict_recursive,
    # qui réutilise le même `tail` d'une itération à l'autre) : la retirer avant
    # de fusionner, sinon merge la renomme en _x/_y au lieu de l'écraser.
    df = df.drop(columns=["cluster_family_roll_mean_28"], errors="ignore").merge(
        cluster_family_daily[["cluster", "family", "date", "cluster_family_roll_mean_28"]],
        on=["cluster", "family", "date"], how="left",
    )

    # feature de volume historique : combien de jours d'historique réel
    # (non-NaN) l'entité a déjà accumulés à cette date — distingue une vraie
    # série à faible volume d'une entité tout juste ouverte (cf. journal).
    df["history_days"] = grouped.cumcount()

    return df


def _build_code_maps(cols: list[str], frame: pd.DataFrame) -> dict[str, dict]:
    maps = {}
    for col in cols:
        values = sorted(frame[col].dropna().unique())
        maps[col] = {v: i for i, v in enumerate(values)}
    return maps


def _apply_code_maps(df: pd.DataFrame, maps: dict[str, dict]) -> None:
    for col, mapping in maps.items():
        df[f"{col}__code"] = df[col].map(mapping).fillna(-1).astype("int32")


def prepare_entities(
    train: pd.DataFrame, stores: pd.DataFrame, oil: pd.DataFrame, holidays: pd.DataFrame, transactions: pd.DataFrame
):
    """Grille complète (store_nbr x family x date), jointe aux exogènes
    globales — même structure de base pour l'historique et pour l'horizon
    futur (holiday/oil/calendaire ne dépendent pas de la cible). `cutoff` pour
    transactions.csv (jamais dispo au-delà) = `train["date"].max()` : le dernier
    jour du `train` reçu par l'appelant, qui est déjà tronqué au bon endroit
    selon le mode (backtest vs vraie soumission), cf. build_transactions_features.
    """
    all_dates = pd.date_range(train["date"].min(), train["date"].max() + pd.Timedelta(days=HORIZON), freq="D")
    oil_series = build_oil_series(oil, all_dates)
    holiday_flags = build_holiday_flags(holidays, stores, all_dates)
    tx_features = build_transactions_features(transactions, all_dates, stores, train["date"].max())

    families = sorted(train["family"].unique())
    grid = pd.MultiIndex.from_product(
        [stores["store_nbr"], families, all_dates], names=["store_nbr", "family", "date"]
    ).to_frame(index=False)
    grid = grid.merge(stores, on="store_nbr", how="left")
    grid = grid.merge(holiday_flags, on=["store_nbr", "date"], how="left")
    grid = grid.merge(tx_features, on=["store_nbr", "date"], how="left")
    grid["oil_price"] = grid["date"].map(oil_series)
    grid = grid.merge(
        train[["store_nbr", "family", "date", "sales", "onpromotion"]],
        on=["store_nbr", "family", "date"],
        how="left",
    )
    grid["onpromotion"] = grid["onpromotion"].fillna(0)

    return grid


def train_and_predict(history: pd.DataFrame, future: pd.DataFrame, valid_days: int = 0):
    code_maps = _build_code_maps(["store_nbr", "family", "city", "state", "type"], history)
    _apply_code_maps(history, code_maps)
    _apply_code_maps(future, code_maps)
    future["cluster"] = future["cluster"].astype("int32")
    history["cluster"] = history["cluster"].astype("int32")

    non_feature = {"date", "sales", "y", "store_nbr", "family", "city", "state", "type"}
    feature_cols = [c for c in history.columns if c not in non_feature]

    if valid_days:
        cutoff = history["date"].max() - pd.Timedelta(days=valid_days - 1)
        fit_df, valid_df = history[history["date"] < cutoff], history[history["date"] >= cutoff]
        train_set = lgb.Dataset(fit_df[feature_cols], label=fit_df["sales"], categorical_feature=CATEGORICAL_COLS)
        valid_set = lgb.Dataset(
            valid_df[feature_cols], label=valid_df["sales"], categorical_feature=CATEGORICAL_COLS, reference=train_set
        )
        booster = lgb.train(
            LGB_PARAMS,
            train_set,
            num_boost_round=NUM_BOOST_ROUND,
            valid_sets=[valid_set],
            callbacks=[lgb.early_stopping(EARLY_STOPPING_ROUNDS), lgb.log_evaluation(100)],
        )
        print(f"  meilleure itération : {booster.best_iteration}")
    else:
        train_set = lgb.Dataset(history[feature_cols], label=history["sales"], categorical_feature=CATEGORICAL_COLS)
        booster = lgb.train(LGB_PARAMS, train_set, num_boost_round=FINAL_NUM_BOOST_ROUND)

    predictions = predict_recursive(booster, history, future, feature_cols)
    return predictions


def predict_recursive(booster: lgb.Booster, history: pd.DataFrame, future: pd.DataFrame, feature_cols: list[str]):
    group_cols = ["store_nbr", "family"]
    tail = history.sort_values([*group_cols, "date"]).reset_index(drop=True)
    rank_from_end = tail.groupby(group_cols, observed=True).cumcount(ascending=False)
    tail = tail[rank_from_end < TAIL_WINDOW_DAYS].reset_index(drop=True)

    future_dates = sorted(future["date"].unique())
    results = []

    for target_date in future_dates:
        day_exog = future[future["date"] == target_date].reset_index(drop=True)
        day_exog["sales"] = np.nan

        combined = pd.concat([tail, day_exog], ignore_index=True, sort=False)
        combined = add_calendar_features(combined)
        combined = add_lag_features(combined, group_cols)

        today_rows = combined.loc[combined["date"] == target_date].reset_index(drop=True)
        # tweedie prédit directement l'échelle des ventes (pas de log-link à inverser)
        pred_sales = np.clip(booster.predict(today_rows[feature_cols]), 0, None)

        # NB : un blend post-hoc avec l'heuristique saisonnière x croissance
        # (yoy_extrapolation) a été testé ici, en universel puis gaté sur
        # |yoy_trend| — dans les deux cas le RMSLE global a nettement empiré
        # (0.49 -> 0.72), l'heuristique étant trop bruitée hors du cas
        # pathologique visé (SCHOOL AND OFFICE SUPPLIES, rentrée scolaire).
        # yoy_extrapolation reste comme feature d'entrée (add_lag_features),
        # laissée à l'arbre, sans correction forcée en sortie.

        day_result = today_rows[[*group_cols, "date"]].copy()
        day_result["prevision"] = pred_sales
        results.append(day_result)

        predicted_rows = today_rows[tail.columns].copy()
        predicted_rows["sales"] = pred_sales
        tail = pd.concat([tail, predicted_rows], ignore_index=True, sort=False)
        rank_from_end = tail.groupby(group_cols, observed=True).cumcount(ascending=False)
        tail = tail[rank_from_end < TAIL_WINDOW_DAYS].reset_index(drop=True)

    return pd.concat(results, ignore_index=True)


def rmsle(actual: np.ndarray, predicted: np.ndarray) -> float:
    actual = np.clip(actual, 0, None)
    predicted = np.clip(predicted, 0, None)
    return float(np.sqrt(np.mean((np.log1p(predicted) - np.log1p(actual)) ** 2)))


def run_validate(data: dict[str, pd.DataFrame]) -> None:
    train, stores, oil, holidays, transactions = (
        data["train"], data["stores"], data["oil"], data["holidays"], data["transactions"]
    )

    cutoff = train["date"].max() - pd.Timedelta(days=HORIZON - 1)
    fit_train = train[train["date"] < cutoff]
    truth = train[train["date"] >= cutoff][["store_nbr", "family", "date", "sales"]]

    print(f"backtest : entraînement jusqu'au {fit_train['date'].max().date()}, "
          f"validation {truth['date'].min().date()} -> {truth['date'].max().date()} ({HORIZON} jours)")

    grid = prepare_entities(fit_train, stores, oil, holidays, transactions)
    history = add_lag_features(add_calendar_features(grid[grid["date"] <= fit_train["date"].max()].copy()), ["store_nbr", "family"])
    future = grid[grid["date"] > fit_train["date"].max()].copy()

    start = time.time()
    predictions = train_and_predict(history, future, valid_days=HORIZON)
    print(f"terminé en {time.time() - start:.0f}s")

    merged = truth.merge(predictions, on=["store_nbr", "family", "date"], how="left")
    merged["prevision"] = merged["prevision"].fillna(0)
    score = rmsle(merged["sales"].to_numpy(), merged["prevision"].to_numpy())
    print(f"RMSLE local (16 derniers jours de train.csv, connu) : {score:.5f}  (SARIMAX : 0.42726)")

    merged["sq_err"] = (np.log1p(merged["prevision"].clip(lower=0)) - np.log1p(merged["sales"].clip(lower=0))) ** 2
    by_family = merged.groupby("family")["sq_err"].mean().pow(0.5).sort_values(ascending=False)
    print("\nRMSLE par famille (10 pires) :")
    print(by_family.head(10).to_string())

    history_days = history.groupby(["store_nbr", "family"], observed=True)["date"].transform("count")
    hist_by_entity = history.groupby(["store_nbr", "family"], observed=True).size().rename("n_days_history")
    merged = merged.merge(hist_by_entity, on=["store_nbr", "family"], how="left")
    merged["bucket"] = pd.cut(merged["n_days_history"], [0, 90, 365, 1000, 100000], labels=["<90j", "90-365j", "1-3ans", ">3ans"])
    by_hist = merged.groupby("bucket", observed=True)["sq_err"].mean().pow(0.5)
    print("\nRMSLE par ancienneté d'historique de l'entité :")
    print(by_hist.to_string())

    out_path = f"{DATA_DIR}/validate_predictions_debug.csv"
    merged.to_csv(out_path, index=False)
    print(f"\ndétail écrit dans {out_path}")


def run_submit(data: dict[str, pd.DataFrame]) -> None:
    train, test, stores, oil, holidays, transactions = (
        data["train"], data["test"], data["stores"], data["oil"], data["holidays"], data["transactions"]
    )

    grid = prepare_entities(train, stores, oil, holidays, transactions)
    history = add_lag_features(add_calendar_features(grid[grid["date"] <= train["date"].max()].copy()), ["store_nbr", "family"])
    future = grid[grid["date"] > train["date"].max()].copy()

    start = time.time()
    predictions = train_and_predict(history, future, valid_days=0)
    print(f"terminé en {time.time() - start:.0f}s")

    result = test[["id", "store_nbr", "family", "date"]].merge(
        predictions, on=["store_nbr", "family", "date"], how="left"
    )
    result["prevision"] = result["prevision"].fillna(0)
    result = result[["id"]].assign(sales=result["prevision"].round(2))

    out_path = f"{DATA_DIR}/store_sales_submission_lightgbm.csv"
    result.to_csv(out_path, index=False)
    print(f"{out_path} ({len(result)} lignes) — à uploader sur Kaggle pour le score officiel")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--submit", action="store_true", help="entraîne sur tout l'historique et écrit le CSV Kaggle")
    parser.add_argument(
        "--family", default=None,
        help="restreint train/test à une seule famille (ex. 'SCHOOL AND OFFICE SUPPLIES') — "
             "pour tester un modèle dédié plutôt que le modèle global sur les 33 familles",
    )
    args = parser.parse_args()

    data = load_raw()
    if args.family:
        data["train"] = data["train"][data["train"]["family"] == args.family].copy()
        data["test"] = data["test"][data["test"]["family"] == args.family].copy()

    if args.submit:
        run_submit(data)
    else:
        run_validate(data)


if __name__ == "__main__":
    main()
