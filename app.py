from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf

from backtest import BacktestConfig, STRATEGY_LABELS, equity_curve, run_sma_sweep


st.set_page_config(page_title="Laboratoire SMA — Apple", page_icon="🍎", layout="wide")

PERIODS = {
    "Depuis 2000": "2000-01-01",
    "Depuis 2010": "2010-01-01",
    "Depuis 2020": "2020-01-01",
}


@st.cache_data(ttl=3600, show_spinner=False)
def download_prices(ticker: str) -> pd.DataFrame:
    data = yf.download(
        ticker,
        start="1999-01-01",
        auto_adjust=True,
        progress=False,
        actions=False,
    )
    if data.empty:
        raise ValueError(f"Aucune donnée reçue pour {ticker}.")
    if isinstance(data.columns, pd.MultiIndex):
        data = data.xs(ticker, axis=1, level=-1)
    return data


def percent(value: float) -> str:
    return "—" if pd.isna(value) else f"{value:.1%}"


st.title("🍎 Laboratoire SMA — Apple")
st.caption(
    "Comparer les croisements sous et au-dessus des SMA 150 à 250, avec exécution "
    "à l'ouverture suivante et comparaison à l'achat-conservation."
)

with st.sidebar:
    st.header("Paramètres")
    ticker = st.text_input("Symbole", value="AAPL").strip().upper()
    selected_periods = st.multiselect(
        "Périodes", list(PERIODS), default=list(PERIODS)
    )
    sma_min, sma_max = st.slider("Plage de SMA", 50, 300, (150, 250))
    initial_capital = st.number_input(
        "Capital initial ($)", min_value=1_000, value=10_000, step=1_000
    )
    cost_bps = st.number_input(
        "Frais + slippage par transaction (points de base)",
        min_value=0.0,
        value=2.0,
        step=0.5,
        help="2 pb = 0,02 % à l'achat et 0,02 % à la vente.",
    )
    cooldown_days = st.number_input(
        "Délai entre deux entrées (jours civils)",
        min_value=0,
        value=30,
        step=1,
        help=(
            "Après une entrée, aucun autre achat n'est permis pendant ce délai. "
            "Une sortie demeure toujours possible."
        ),
    )
    min_trades = st.number_input(
        "Nombre minimal de transactions", min_value=0, value=3, step=1
    )
    run = st.button("Lancer le backtest", type="primary", use_container_width=True)

if not selected_periods:
    st.warning("Choisis au moins une période dans le menu de gauche.")
    st.stop()

if run or "results" not in st.session_state:
    try:
        with st.spinner("Téléchargement des prix et calcul des 606 scénarios…"):
            prices = download_prices(ticker)
            config = BacktestConfig(
                float(initial_capital), float(cost_bps), int(cooldown_days)
            )
            results = run_sma_sweep(
                prices,
                {label: PERIODS[label] for label in selected_periods},
                range(sma_min, sma_max + 1),
                config=config,
            )
            st.session_state.update(
                results=results,
                prices=prices,
                config=config,
                ticker=ticker,
                periods=selected_periods,
            )
    except Exception as exc:
        st.error(f"Le backtest n'a pas pu être exécuté : {exc}")
        st.stop()

results = st.session_state["results"]
prices = st.session_state["prices"]
config = st.session_state["config"]

eligible = results[results["Transactions"] >= min_trades].copy()
if eligible.empty:
    st.warning("Aucun scénario ne respecte le nombre minimal de transactions.")
    st.stop()

best = (
    eligible.sort_values("Rendement annualisé", ascending=False)
    .groupby(["Période", "Stratégie"], as_index=False)
    .first()
)

st.subheader("Meilleure SMA pour chaque scénario")
cards = st.columns(min(3, len(best)))
for index, (_, row) in enumerate(best.iterrows()):
    with cards[index % len(cards)]:
        st.metric(
            f"{row['Période']} — SMA {int(row['SMA'])}",
            percent(row["Rendement annualisé"]),
            percent(row["Surperformance annualisée"]),
            help=f"{row['Stratégie']} — l'écart affiché est la surperformance annualisée.",
        )

tab_rank, tab_heatmap, tab_curve, tab_method = st.tabs(
    ["Classement", "Profil des rendements", "Courbe détaillée", "Méthodologie"]
)

with tab_rank:
    period_filter = st.selectbox("Période du classement", selected_periods)
    strategy_filter = st.selectbox("Stratégie", list(STRATEGY_LABELS.values()))
    ranking = eligible[
        (eligible["Période"] == period_filter)
        & (eligible["Stratégie"] == strategy_filter)
    ].sort_values("Rendement annualisé", ascending=False)
    display_cols = [
        "SMA", "Rendement total", "Rendement annualisé",
        "Rendement annualisé achat-conservation", "Surperformance annualisée",
        "Drawdown maximal",
        "Transactions", "Taux de réussite", "Exposition", "Capital final",
    ]
    st.dataframe(
        ranking[display_cols].style.format(
            {
                "Rendement total": "{:.1%}",
                "Rendement annualisé": "{:.1%}",
                "Rendement annualisé achat-conservation": "{:.1%}",
                "Surperformance annualisée": "{:.1%}",
                "Drawdown maximal": "{:.1%}",
                "Taux de réussite": "{:.1%}",
                "Exposition": "{:.1%}",
                "Capital final": "${:,.0f}",
            }
        ),
        use_container_width=True,
        hide_index=True,
    )
    st.download_button(
        "Télécharger tous les résultats (CSV)",
        results.to_csv(index=False).encode("utf-8-sig"),
        file_name=f"resultats_sma_{st.session_state['ticker']}.csv",
        mime="text/csv",
    )

with tab_heatmap:
    heat_period = st.selectbox("Période", selected_periods, key="heat_period")
    heat = eligible[eligible["Période"] == heat_period].copy()
    fig = px.line(
        heat,
        x="SMA",
        y="Rendement annualisé",
        color="Stratégie",
        labels={"Rendement annualisé": "Rendement annualisé", "SMA": "Période SMA"},
    )
    fig.update_yaxes(tickformat=".1%")
    fig.update_layout(legend_title_text="")
    st.plotly_chart(fig, use_container_width=True)

with tab_curve:
    curve_period = st.selectbox("Période", selected_periods, key="curve_period")
    curve_strategy_label = st.selectbox(
        "Stratégie", list(STRATEGY_LABELS.values()), key="curve_strategy"
    )
    default_sma = min(max(200, sma_min), sma_max)
    curve_sma = st.slider("SMA à examiner", sma_min, sma_max, default_sma)
    strategy_code = next(
        code for code, label in STRATEGY_LABELS.items() if label == curve_strategy_label
    )
    curve = equity_curve(
        prices,
        curve_sma,
        strategy_code,
        config,
        active_from=PERIODS[curve_period],
    )
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=curve.index, y=curve["Equity"], name="Stratégie"))
    fig.add_trace(
        go.Scatter(x=curve.index, y=curve["Benchmark"], name="Achat-conservation")
    )
    fig.update_layout(yaxis_title="Valeur du portefeuille ($)", legend_title_text="")
    st.plotly_chart(fig, use_container_width=True)

with tab_method:
    st.markdown(
        """
        **Stratégie 1 — croisement sous la SMA :** achat à l'ouverture suivant le
        croisement sous la moyenne; vente à l'ouverture suivant le retour au-dessus.

        **Stratégie 2 — croisement au-dessus de la SMA :** achat à l'ouverture suivant
        le croisement au-dessus; vente à l'ouverture suivant le retour sous la moyenne.

        Les prix sont ajustés pour les fractionnements et dividendes. Le modèle est
        entièrement investi ou en liquidités, sans vente à découvert. Les frais sont
        appliqués à chaque entrée et sortie. Après une entrée, les nouveaux signaux
        d'achat sont bloqués pendant le délai choisi; les sorties restent permises.
        Les résultats passés ne garantissent pas
        les résultats futurs et ce laboratoire ne constitue pas une recommandation.
        """
    )
