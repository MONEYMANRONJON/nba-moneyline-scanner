import streamlit as st
import pandas as pd
import numpy as np
import requests
from datetime import date
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

st.set_page_config(
    page_title="NBA Moneyline Scanner",
    page_icon="🏀",
    layout="centered",
)

REPO_RAW = "https://raw.githubusercontent.com/llimllib/nba_data/main/data/"
SEASONS = [2022, 2023, 2024, 2025, 2026, 2027]
ROLLING_WINDOWS = [5, 10]

MIN_MODEL_PROB = 0.55
MIN_EDGE = 0.04
STAKE = 100.0


def american_to_implied(ml):
    ml = float(ml)
    if ml > 0:
        return 100.0 / (ml + 100.0)
    return abs(ml) / (abs(ml) + 100.0)


def probability_to_american(prob):
    if prob <= 0 or prob >= 1:
        return np.nan
    if prob >= 0.50:
        return -100.0 * prob / (1.0 - prob)
    return 100.0 * (1.0 - prob) / prob


def payout_for_win(stake, ml):
    ml = float(ml)
    if ml > 0:
        return stake * ml / 100.0
    return stake * 100.0 / abs(ml)


def pct(x):
    return f"{x * 100:.2f}%"


@st.cache_data(ttl=21600, show_spinner=False)
def load_data():
    frames = []

    for season in SEASONS:
        url = f"{REPO_RAW}gamelog_{season}.parquet"

        try:
            df = pd.read_parquet(url)
            df["season_year"] = season
            frames.append(df)
        except Exception:
            pass

    if not frames:
        raise RuntimeError("NBA data could not be downloaded.")

    data = pd.concat(frames, ignore_index=True)

    required = [
        "game_id",
        "game_date",
        "team_abbreviation",
        "matchup",
        "wl",
        "off_rating",
        "def_rating",
        "net_rating",
        "pace",
    ]

    missing = [c for c in required if c not in data.columns]

    if missing:
        raise RuntimeError(f"Missing required NBA columns: {missing}")

    data["game_id"] = data["game_id"].astype(str)

    data["game_date"] = pd.to_datetime(
        data["game_date"],
        errors="coerce"
    )

    data = data.dropna(
        subset=[
            "game_id",
            "game_date",
            "team_abbreviation"
        ]
    )

    data = data[
        data["game_id"].str.startswith("002")
    ].copy()

    data = data.drop_duplicates(
        ["game_id", "team_abbreviation"]
    )

    for col in [
        "off_rating",
        "def_rating",
        "net_rating",
        "pace"
    ]:
        data[col] = pd.to_numeric(
            data[col],
            errors="coerce"
        )

    data["win"] = (
        data["wl"]
        .astype(str)
        .str.upper()
        .eq("W")
        .astype(int)
    )

    data = data.sort_values(
        [
            "team_abbreviation",
            "game_date",
            "game_id"
        ]
    ).reset_index(drop=True)

    data["is_home"] = (
        data["matchup"]
        .astype(str)
        .str.contains(
            "vs.",
            regex=False
        )
        .astype(int)
    )

    grouped = data.groupby(
        "team_abbreviation",
        group_keys=False
    )

    data["previous_game_date"] = (
        grouped["game_date"].shift(1)
    )

    data["rest_days"] = (
        data["game_date"]
        - data["previous_game_date"]
    ).dt.days

    data["rest_days"] = (
        data["rest_days"]
        .clip(0, 30)
        .fillna(7)
    )

    data["back_to_back"] = (
        data["rest_days"] <= 1
    ).astype(int)

    for window in ROLLING_WINDOWS:

        for stat in [
            "off_rating",
            "def_rating",
            "net_rating",
            "pace",
            "win"
        ]:

            data[f"{stat}_l{window}"] = (
                grouped[stat]
                .transform(
                    lambda x:
                    x.shift(1)
                    .rolling(
                        window,
                        min_periods=1
                    )
                    .mean()
                )
            )

    keep = [
        "game_id",
        "season_year",
        "game_date",
        "team_abbreviation",
        "wl",
        "rest_days",
        "back_to_back",
    ]

    for window in ROLLING_WINDOWS:

        for stat in [
            "off_rating",
            "def_rating",
            "net_rating",
            "pace",
            "win"
        ]:

            keep.append(
                f"{stat}_l{window}"
            )

    home = data[
        data["is_home"] == 1
    ][keep].copy()

    away = data[
        data["is_home"] == 0
    ][keep].copy()

    home = home.rename(
        columns={
            c: f"home_{c}"
            for c in home.columns
            if c != "game_id"
        }
    )

    away = away.rename(
        columns={
            c: f"away_{c}"
            for c in away.columns
            if c != "game_id"
        }
    )

    games = pd.merge(
        home,
        away,
        on="game_id",
        how="inner"
    )

    games["home_win"] = (
        games["home_wl"]
        .astype(str)
        .str.upper()
        .eq("W")
        .astype(int)
    )

    games["game_date"] = (
        games["home_game_date"]
    )

    games = games.sort_values(
        [
            "game_date",
            "game_id"
        ]
    ).reset_index(drop=True)

    feature_names = []

    for window in ROLLING_WINDOWS:

        for stat in [
            "off_rating",
            "def_rating",
            "net_rating",
            "pace",
            "win"
        ]:

            name = (
                f"diff_{stat}_l{window}"
            )

            games[name] = (
                games[
                    f"home_{stat}_l{window}"
                ]
                -
                games[
                    f"away_{stat}_l{window}"
                ]
            )

            feature_names.append(name)

    games["diff_rest"] = (
        games["home_rest_days"]
        -
        games["away_rest_days"]
    )

    games["home_b2b"] = (
        games["home_back_to_back"]
    )

    games["away_b2b"] = (
        games["away_back_to_back"]
    )

    games["diff_b2b"] = (
        games["away_b2b"]
        -
        games["home_b2b"]
    )

    feature_names += [
        "diff_rest",
        "home_b2b",
        "away_b2b",
        "diff_b2b"
    ]

    model_games = games.dropna(
        subset=feature_names + ["home_win"]
    ).copy()

    X = model_games[
        feature_names
    ]

    y = model_games[
        "home_win"
    ].astype(int)

    model = Pipeline(
        [
            (
                "imputer",
                SimpleImputer(
                    strategy="median"
                )
            ),
            (
                "classifier",
                HistGradientBoostingClassifier(
                    max_iter=300,
                    learning_rate=0.04,
                    max_leaf_nodes=15,
                    l2_regularization=1.0,
                    random_state=42
                )
            )
        ]
    )

    model.fit(X, y)

    team_history = {}

    for team, group in data.groupby(
        "team_abbreviation"
    ):

        group = group.sort_values(
            [
                "game_date",
                "game_id"
            ]
        )

        recent = group.tail(10)

        if len(recent) == 0:
            continue

        team_history[team] = {
            "off_rating_l5":
                recent["off_rating"]
                .tail(5)
                .mean(),

            "def_rating_l5":
                recent["def_rating"]
                .tail(5)
                .mean(),

            "net_rating_l5":
                recent["net_rating"]
                .tail(5)
                .mean(),

            "pace_l5":
                recent["pace"]
                .tail(5)
                .mean(),

            "win_l5":
                recent["win"]
                .tail(5)
                .mean(),

            "off_rating_l10":
                recent["off_rating"]
                .mean(),

            "def_rating_l10":
                recent["def_rating"]
                .mean(),

            "net_rating_l10":
                recent["net_rating"]
                .mean(),

            "pace_l10":
                recent["pace"]
                .mean(),

            "win_l10":
                recent["win"]
                .mean(),
        }

    return (
        model,
        feature_names,
        team_history,
        len(model_games),
        data["game_date"].max()
    )


def is_regular_season(event):

    season = event.get(
        "season",
        {}
    )

    value = (
        season.get("type")
        if isinstance(season, dict)
        else None
    )

    if isinstance(value, int):
        return value == 2

    if isinstance(value, str):
        return value.lower() in [
            "2",
            "regular",
            "regular season",
            "regular-season"
        ]

    if isinstance(value, dict):

        name = str(
            value.get(
                "name",
                ""
            )
        ).lower()

        abbr = str(
            value.get(
                "abbreviation",
                ""
            )
        ).lower()

        return (
            name in [
                "regular season",
                "regular"
            ]
            or
            abbr in [
                "reg",
                "regular"
            ]
        )

    value = event.get(
        "seasonType"
    )

    if isinstance(value, int):
        return value == 2

    if isinstance(value, str):
        return value.lower() in [
            "2",
            "regular",
            "regular season"
        ]

    return (
        "preseason"
        not in str(
            event.get(
                "name",
                ""
            )
        ).lower()
    )


@st.cache_data(
    ttl=300,
    show_spinner=False
)
def get_games_for_date(
    date_string
):

    url = (
        "https://site.api.espn.com/apis/site/v2/"
        "sports/basketball/nba/scoreboard"
        f"?dates={date_string}"
    )

    response = requests.get(
        url,
        timeout=20
    )

    response.raise_for_status()

    schedule = response.json()

    games = []

    for event in schedule.get(
        "events",
        []
    ):

        if not is_regular_season(
            event
        ):
            continue

        competitions = event.get(
            "competitions",
            []
        )

        if not competitions:
            continue

        competitors = competitions[
            0
        ].get(
            "competitors",
            []
        )

        home = None
        away = None

        for c in competitors:

            abbr = (
                c.get(
                    "team",
                    {}
                )
                .get(
                    "abbreviation"
                )
            )

            if c.get(
                "homeAway"
            ) == "home":

                home = abbr

            elif c.get(
                "homeAway"
            ) == "away":

                away = abbr

        if home and away:

            games.append(
                {
                    "id": event.get(
                        "id"
                    ),
                    "away": away,
                    "home": home,
                    "name": event.get(
                        "name",
                        f"{away} @ {home}"
                    )
                }
            )

    return games


def predict_game(
    model,
    feature_names,
    team_history,
    away,
    home
):

    if (
        away not in team_history
        or
        home not in team_history
    ):
        return None

    a = team_history[away]
    h = team_history[home]

    row = {}

    for window in ROLLING_WINDOWS:

        for stat in [
            "off_rating",
            "def_rating",
            "net_rating",
            "pace",
            "win"
        ]:

            row[
                f"diff_{stat}_l{window}"
            ] = (
                h[
                    f"{stat}_l{window}"
                ]
                -
                a[
                    f"{stat}_l{window}"
                ]
            )

    row["diff_rest"] = 0.0
    row["home_b2b"] = 0
    row["away_b2b"] = 0
    row["diff_b2b"] = 0

    X = pd.DataFrame(
        [row]
    )[feature_names]

    home_prob = float(
        model.predict_proba(
            X
        )[0, 1]
    )

    return home_prob


st.title(
    "🏀 NBA Moneyline Scanner"
)

st.caption(
    "Moneyline only • $100 flat stake • Hard Rock pricing"
)

with st.sidebar:

    st.header(
        "Locked Rules"
    )

    st.write(
        "Minimum model probability: **55%**"
    )

    st.write(
        "Minimum edge: **4%**"
    )

    st.write(
        "Stake: **$100 flat**"
    )

    st.write(
        "Market: **NBA Moneyline only**"
    )

    st.divider()

    st.caption(
        "Historical model data is used "
        "to generate probabilities. "
        "Enter the actual Hard Rock price "
        "available to you."
    )


try:

    with st.spinner(
        "Loading and training the NBA model..."
    ):

        (
            model,
            feature_names,
            team_history,
            training_games,
            latest_data
        ) = load_data()

    st.success(
        "Model ready"
    )

    st.caption(
        f"Training games: {training_games:,} • "
        f"Latest source data: {latest_data.date()}"
    )

except Exception as e:

    st.error(
        f"Could not initialize the model: {e}"
    )

    st.stop()


selected_date = st.date_input(
    "NBA date",
    value=date.today()
)


if st.button(
    "Scan NBA Games",
    type="primary",
    use_container_width=True
):

    date_string = (
        selected_date.strftime(
            "%Y%m%d"
        )
    )

    try:

        with st.spinner(
            "Checking the NBA schedule..."
        ):

            games = get_games_for_date(
                date_string
            )

    except Exception as e:

        st.error(
            f"Could not retrieve the NBA schedule: {e}"
        )

        st.stop()

    if not games:

        st.info(
            "No qualifying regular-season "
            "NBA games found for this date."
        )

        st.stop()

    st.subheader(
        f"{len(games)} game(s) found"
    )

    for game in games:

        away = game["away"]
        home = game["home"]

        st.markdown("---")

        st.subheader(
            f"{away} @ {home}"
        )

        home_prob = predict_game(
            model,
            feature_names,
            team_history,
            away,
            home
        )

        if home_prob is None:

            st.warning(
                "Insufficient team history "
                "for this matchup."
            )

            continue

        away_prob = 1.0 - home_prob

        col1, col2 = st.columns(2)

        with col1:

            st.metric(
                f"{home} model probability",
                pct(home_prob)
            )

        with col2:

            st.metric(
                f"{away} model probability",
                pct(away_prob)
            )

        c1, c2 = st.columns(2)

        with c1:

            home_ml = st.number_input(
                f"{home} Hard Rock ML",
                value=0,
                step=5,
                key=f"home_{game['id']}"
            )

        with c2:

            away_ml = st.number_input(
                f"{away} Hard Rock ML",
                value=0,
                step=5,
                key=f"away_{game['id']}"
            )

        if (
            home_ml == 0
            or
            away_ml == 0
        ):

            st.caption(
                "Enter both Hard Rock moneylines "
                "to calculate the signal."
            )

            continue

        home_imp = american_to_implied(
            home_ml
        )

        away_imp = american_to_implied(
            away_ml
        )

        total_imp = (
            home_imp
            +
            away_imp
        )

        home_market = (
            home_imp
            /
            total_imp
        )

        away_market = (
            away_imp
            /
            total_imp
        )

        home_edge = (
            home_prob
            -
            home_market
        )

        away_edge = (
            away_prob
            -
            away_market
        )

        if home_edge >= away_edge:

            selected_team = home
            selected_ml = home_ml
            selected_prob = home_prob
            selected_market = home_market
            selected_edge = home_edge

        else:

            selected_team = away
            selected_ml = away_ml
            selected_prob = away_prob
            selected_market = away_market
            selected_edge = away_edge

        profit_if_win = payout_for_win(
            STAKE,
            selected_ml
        )

        expected_profit = (
            selected_prob
            *
            profit_if_win
            -
            (
                1 - selected_prob
            )
            *
            STAKE
        )

        ev = (
            expected_profit
            /
            STAKE
        )

        fair_ml = probability_to_american(
            selected_prob
        )

        qualifies = (
            selected_prob
            >= MIN_MODEL_PROB
            and
            selected_edge
            >= MIN_EDGE
        )

        st.markdown(
            "### Signal"
        )

        r1, r2, r3 = st.columns(3)

        with r1:

            st.metric(
                "Selected side",
                selected_team
            )

        with r2:

            st.metric(
                "Model probability",
                pct(selected_prob)
            )

        with r3:

            st.metric(
                "Model edge",
                pct(selected_edge)
            )

        r4, r5, r6 = st.columns(3)

        with r4:

            st.metric(
                "No-vig market",
                pct(selected_market)
            )

        with r5:

            st.metric(
                "Fair ML",
                f"{fair_ml:+.0f}"
            )

        with r6:

            st.metric(
                "EV",
                f"{ev * 100:.2f}%"
            )

        if qualifies:

            st.success(
                f"BET — {selected_team} at "
                f"{selected_ml:+.0f} | "
                f"$100 flat stake | "
                f"Expected profit "
                f"${expected_profit:.2f}"
            )

        else:

            st.error(
                "PASS — locked betting criteria "
                "not met."
            )


st.divider()

st.caption(
    "This tool is a decision-support model, "
    "not a guarantee of profit. "
    "Use the actual Hard Rock price available "
    "at the time of the wager."
)
