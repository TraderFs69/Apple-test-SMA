from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st
import yfinance as yf

from backtest import (
    HORIZON_ORDER,
    STRATEGY_LABELS,
    StudyConfig,
    run_event_study,
    summarize_events,
)


st.set_page_config(page_title="Étude SMA — Apple", page_icon="🍎", layout="wide")

PERIODS = {
    "Depuis 2000": "2000-01-01",
    "Depuis 2010": "2010-01-01",
    "Depuis 2020": "2020-01-01",
}


@st.cache_data(ttl=3600, show_spinner=False)
def download_prices(ticker: str) -> pd.DataFrame:
    data = yf.download(
        ticker,
        start="1998-01-01",
        auto_adjust=True,
        progress=False,
        actions=False,
    )
    if data.empty:
        raise ValueError(f"Aucune donnée reçue pour {ticker}.")
    if isinstance(data.columns, pd.MultiIndex):
        data = data.xs(ticker, axis=1, level=-1)
    return data


st.title("🍎 Étude des croisements SMA — Apple")
st.caption(
    "Aucune règle de vente : chaque signal est une observation indépendante du "
    "rendement futur d'Apple."
)

with st.sidebar:
    st.header("Paramètres")
    ticker = st.text_input("Symbole", value="AAPL").strip().upper()
    selected_periods = st.multiselect(
        "Périodes", list(PERIODS), default=list(PERIODS)
    )
    sma_min, sma_max = st.slider("Plage de SMA", 50, 300, (150, 250))
    cooldown_days = st.number_input(
        "Blocage après un signal (jours civils)",
        min_value=0,
        value=30,
        step=1,
        help="Les nouveaux signaux de la même SMA et direction sont ignorés pendant ce délai.",
    )
    min_observations = st.number_input(
        "Observations minimales", min_value=1, value=3, step=1
    )
    run = st.button("Lancer l'étude", type="primary", use_container_width=True)

if not selected_periods:
    st.warning("Choisis au moins une période.")
    st.stop()

if run or "events" not in st.session_state:
    try:
        with st.spinner("Calcul des signaux et de tous les rendements futurs…"):
            prices = download_prices(ticker)
            events = run_event_study(
                prices,
                {label: PERIODS[label] for label in selected_periods},
                range(sma_min, sma_max + 1),
                config=StudyConfig(int(cooldown_days)),
            )
            st.session_state.update(
                events=events,
                summary=summarize_events(events),
                ticker=ticker,
                periods=selected_periods,
            )
    except Exception as exc:
        st.error(f"L'étude n'a pas pu être exécutée : {exc}")
        st.stop()

events = st.session_state["events"]
summary = st.session_state["summary"]
available_horizons = [h for h in HORIZON_ORDER if h in set(events["Horizon"].astype(str))]

top_horizon = st.selectbox(
    "Horizon à analyser", available_horizons, index=min(4, len(available_horizons) - 1)
)
eligible = summary[
    (summary["Horizon"].astype(str) == top_horizon)
    & (summary["Observations"] >= min_observations)
].copy()

if eligible.empty:
    st.warning("Aucune SMA ne respecte le nombre minimal d'observations à cet horizon.")
    st.stop()

best = (
    eligible.sort_values("Rendement médian", ascending=False)
    .groupby(["Période", "Stratégie"], as_index=False)
    .first()
)

st.subheader(f"Meilleures SMA après {top_horizon}")
columns = st.columns(min(3, len(best)))
for index, (_, row) in enumerate(best.iterrows()):
    with columns[index % len(columns)]:
        st.metric(
            f"{row['Période']} — SMA {int(row['SMA'])}",
            f"{row['Rendement médian']:.1%}",
            f"{int(row['Observations'])} observations",
            help=f"{row['Stratégie']} — rendement médian affiché.",
        )

tab_rank, tab_profile, tab_events, tab_method = st.tabs(
    ["Classement", "Profil par SMA", "Signaux individuels", "Méthodologie"]
)

with tab_rank:
    rank_period = st.selectbox("Période", selected_periods, key="rank_period")
    rank_strategy = st.selectbox(
        "Direction", list(STRATEGY_LABELS.values()), key="rank_strategy"
    )
    ranking = eligible[
        (eligible["Période"] == rank_period)
        & (eligible["Stratégie"] == rank_strategy)
    ].sort_values("Rendement médian", ascending=False)
    cols = [
        "SMA", "Observations", "Rendement moyen", "Rendement médian",
        "Taux positif", "Meilleur rendement", "Pire rendement", "Écart-type",
    ]
    st.dataframe(
        ranking[cols].style.format(
            {
                "Rendement moyen": "{:.1%}",
                "Rendement médian": "{:.1%}",
                "Taux positif": "{:.1%}",
                "Meilleur rendement": "{:.1%}",
                "Pire rendement": "{:.1%}",
                "Écart-type": "{:.1%}",
            }
        ),
        use_container_width=True,
        hide_index=True,
    )

with tab_profile:
    profile_period = st.selectbox("Période", selected_periods, key="profile_period")
    profile = eligible[eligible["Période"] == profile_period]
    fig = px.line(
        profile,
        x="SMA",
        y="Rendement médian",
        color="Stratégie",
        labels={"Rendement médian": "Rendement médian", "SMA": "Période SMA"},
    )
    fig.update_yaxes(tickformat=".1%")
    fig.update_layout(legend_title_text="")
    st.plotly_chart(fig, use_container_width=True)

with tab_events:
    event_period = st.selectbox("Période", selected_periods, key="event_period")
    event_strategy = st.selectbox(
        "Direction", list(STRATEGY_LABELS.values()), key="event_strategy"
    )
    default_sma = min(max(200, sma_min), sma_max)
    event_sma = st.slider("SMA", sma_min, sma_max, default_sma, key="event_sma")
    detail = events[
        (events["Période"] == event_period)
        & (events["Stratégie"] == event_strategy)
        & (events["SMA"] == event_sma)
        & (events["Horizon"].astype(str) == top_horizon)
    ].sort_values("Date du signal", ascending=False)
    st.dataframe(
        detail[
            [
                "Date du signal", "Date d'entrée", "Prix d'entrée",
                "Date d'observation", "Prix d'observation", "Rendement",
            ]
        ].style.format(
            {
                "Prix d'entrée": "${:.2f}",
                "Prix d'observation": "${:.2f}",
                "Rendement": "{:.1%}",
            }
        ),
        use_container_width=True,
        hide_index=True,
    )
    st.download_button(
        "Télécharger toutes les observations (CSV)",
        events.to_csv(index=False).encode("utf-8-sig"),
        file_name=f"etude_evenements_sma_{st.session_state['ticker']}.csv",
        mime="text/csv",
    )

with tab_method:
    st.markdown(
        """
        **Signal 1 — croisement sous la SMA :** le cours clôture sous la moyenne après
        avoir clôturé au-dessus la veille.

        **Signal 2 — croisement au-dessus de la SMA :** le cours clôture au-dessus de
        la moyenne après avoir clôturé sous celle-ci la veille.

        Le prix de référence est l'ouverture de la séance suivant le signal. Les horizons
        de 10 à 200 jours correspondent à des séances boursières. Les horizons de 1 à
        10 ans utilisent la première séance disponible à la date anniversaire. Il n'y a
        aucune règle de vente : on mesure uniquement le rendement à chaque horizon.

        Après un signal accepté, les signaux suivants de la même SMA et de la même
        direction sont ignorés pendant 30 jours civils, ou pendant le délai choisi.
        Les prix sont ajustés pour les fractionnements et dividendes. Une observation
        sans historique futur suffisant est simplement exclue de l'horizon concerné.
        """
    )


