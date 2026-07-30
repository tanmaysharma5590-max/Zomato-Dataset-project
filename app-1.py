"""
Tandoor - Zomato Restaurant Intelligence
A single-file Streamlit app: EDA, feature engineering, and a live rating
prediction model, built on the Zomato restaurant dataset.

Run with:
    pip install -r requirements.txt
    streamlit run app.py

Pages (sidebar): Overview, Explore the Data, Predict a Rating, Insights.
"""

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

st.set_page_config(
    page_title="Tandoor · Zomato Restaurant Intelligence",
    page_icon="🍽️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Theme ──────────────────────────────────────────────────────────────
BG = "#1B1310"
PANEL = "#241A15"
BORDER = "#3A2A20"
TEXT = "#F1E7DD"
MUTED = "#A8968A"
ACCENT = "#D98A4F"
GOLD = "#E8A33D"
GOOD = "#3FA66B"
BAR_A, BAR_B = "#CBB69A", "#A85F36"
PLOTLY_TEMPLATE = "plotly_dark"

st.markdown(f"""
<style>
.stApp {{ background-color: {BG}; color: {TEXT}; }}
section[data-testid="stSidebar"] {{ background-color: #150F0C; border-right: 1px solid {BORDER}; }}
h1, h2, h3, h4, p, span, label {{ color: {TEXT}; }}
.brand-title {{ color: {ACCENT}; font-size: 1.25rem; font-weight: 800; margin-bottom: 0; }}
.brand-sub {{ color: {MUTED}; font-size: 0.72rem; text-transform: uppercase; margin-top: -4px; }}
.sidebar-footer {{ color: {MUTED}; font-size: 0.72rem; border-top: 1px solid {BORDER}; padding-top: 0.7rem; margin-top: 1rem; }}
.panel {{ background: {PANEL}; border: 1px solid {BORDER}; border-radius: 12px; padding: 1.1rem 1.3rem; }}
.metric-card {{ background: {PANEL}; border: 1px solid {BORDER}; border-radius: 10px; padding: 0.9rem 1rem; text-align: center; }}
.metric-card .value {{ font-size: 1.6rem; font-weight: 800; color: {ACCENT}; }}
.metric-card .label {{ font-size: 0.78rem; color: {MUTED}; margin-top: 2px; }}
.kicker {{ color: {ACCENT}; font-size: 0.75rem; font-weight: 700; text-transform: uppercase; }}
.badge {{ display: inline-block; border-radius: 6px; padding: 0.35rem 0.7rem; font-size: 0.82rem; font-weight: 600; }}
.stButton > button {{ background-color: {GOLD}; color: #241A15; font-weight: 700; border: none; border-radius: 8px; }}
.footer-note {{ text-align: center; color: {MUTED}; font-size: 0.75rem; margin-top: 2rem; }}
</style>
""", unsafe_allow_html=True)


# ── Data + model ───────────────────────────────────────────────────────
@st.cache_data
def load_default_data():
    df = pd.read_csv("data/zomato.csv", encoding="latin-1")
    country = pd.read_excel("data/Country-Code.xlsx")
    return pd.merge(df, country, on="Country Code", how="left")


@st.cache_resource
def train_model(df: pd.DataFrame):
    """Random Forest predicting Aggregate rating from a restaurant's profile."""
    model_df = df[df["Aggregate rating"] > 0].copy()

    country_encoder = LabelEncoder()
    model_df["Country_enc"] = country_encoder.fit_transform(model_df["Country"].astype(str))
    model_df["Table_enc"] = (model_df["Has Table booking"] == "Yes").astype(int)
    model_df["Delivery_enc"] = (model_df["Has Online delivery"] == "Yes").astype(int)

    feature_cols = ["Average Cost for two", "Votes", "Price range",
                     "Delivery_enc", "Table_enc", "Country_enc"]
    X, y = model_df[feature_cols], model_df["Aggregate rating"]

    rf = RandomForestRegressor(n_estimators=150, max_depth=10, random_state=42, n_jobs=-1)
    rf.fit(X, y)
    return rf, country_encoder, feature_cols


def rating_badge(rating: float):
    if rating >= 4.5:
        return "Excellent — top-tier, exceptional feedback expected.", "#2E8B57"
    if rating >= 4.0:
        return "Very good — strongly liked by most customers.", "#4CA35A"
    if rating >= 3.5:
        return "Good — a solid, well-regarded rating range.", GOOD
    if rating >= 2.5:
        return "Average — room to improve on service or value.", GOLD
    return "Below average — likely to need attention.", "#C1543F"


df = load_default_data()

# ── Sidebar ────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<p class="brand-title">🍽️ Tandoor</p>', unsafe_allow_html=True)
    st.markdown('<p class="brand-sub">Zomato Restaurant Intelligence</p>', unsafe_allow_html=True)
    st.write("")
    page = st.radio("Navigate",
        ["🏠  Overview", "🔍  Explore the Data", "⭐  Predict a Rating", "💡  Insights"],
        label_visibility="collapsed")

    model, country_encoder, feature_cols = train_model(df)
    st.markdown(
        f'<div class="sidebar-footer">Dataset: {df.shape[0]:,} restaurants across '
        f'{df["Country"].nunique()} countries<br>Source: Zomato restaurant dataset</div>',
        unsafe_allow_html=True,
    )


# ── Page 1: Overview ──────────────────────────────────────────────────
if page == "🏠  Overview":
    st.markdown('<p class="kicker">Overview</p>', unsafe_allow_html=True)
    st.title("Zomato Dataset — EDA & Feature Engineering")
    st.write("A dashboard covering EDA, feature engineering, and a live rating-prediction model.")

    stats = [
        (f"{df.shape[0]:,}", "Restaurants"),
        (f"{df['Country'].nunique()}", "Countries"),
        (f"{df['City'].nunique()}", "Cities"),
        (f"{df['Cuisines'].nunique():,}", "Cuisine Combos"),
        (f"{df['Aggregate rating'].replace(0, np.nan).mean():.2f} ★", "Avg. Rating (rated only)"),
    ]
    cols = st.columns(5)
    for col, (val, lab) in zip(cols, stats):
        col.markdown(f'<div class="metric-card"><div class="value">{val}</div>'
                      f'<div class="label">{lab}</div></div>', unsafe_allow_html=True)

    st.write("")
    left, right = st.columns([1.3, 1])
    with left:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.subheader("What's inside")
        st.markdown(
            "- **Explore the Data** — how votes, price, and table booking relate to rating.\n"
            "- **Predict a Rating** — a live Random Forest estimates a restaurant's rating."
        )
        st.markdown("</div>", unsafe_allow_html=True)
    with right:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.subheader("Preview")
        st.dataframe(df[["Restaurant Name", "Country", "Aggregate rating", "Votes"]].head(6),
                     use_container_width=True, hide_index=True)
        st.markdown("</div>", unsafe_allow_html=True)


# ── Page 2: Explore the Data ─────────────────────────────────────────
elif page == "🔍  Explore the Data":
    st.markdown('<p class="kicker">Explore</p>', unsafe_allow_html=True)
    st.title("Explore the Data")
    st.caption("Filter by country, then see how votes, price, and service options relate to rating.")

    country_filter = st.multiselect("Filter by country (optional)", sorted(df["Country"].dropna().unique()))
    view = df[df["Country"].isin(country_filter)] if country_filter else df
    rated = view[view["Aggregate rating"] > 0]

    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown("**Votes vs. rating**, bubble size = votes, color = price range")
    fig1 = px.scatter(rated, x="Votes", y="Aggregate rating", size="Votes", color="Price range",
                       color_continuous_scale=["#4a2f1f", "#a85f36", "#d98a4f", "#f2b25a"],
                       log_x=True, opacity=0.75, template=PLOTLY_TEMPLATE)
    fig1.update_layout(paper_bgcolor=PANEL, plot_bgcolor=PANEL, font_color=TEXT,
                        margin=dict(l=10, r=10, t=10, b=10), height=380)
    st.plotly_chart(fig1, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    st.write("")
    c1, c2 = st.columns(2)
    for col, field, label in [(c1, "Has Table booking", "Table booking"), (c2, "Has Online delivery", "Online delivery")]:
        with col:
            st.markdown('<div class="panel">', unsafe_allow_html=True)
            st.markdown(f"**{label} vs. rating**")
            avg = rated.groupby(field)["Aggregate rating"].mean().reindex(["No", "Yes"])
            fig = go.Figure(go.Bar(x=avg.index, y=avg.values, marker_color=[BAR_A, BAR_B]))
            fig.update_layout(template=PLOTLY_TEMPLATE, paper_bgcolor=PANEL, plot_bgcolor=PANEL,
                               font_color=TEXT, yaxis_title="Avg. rating",
                               margin=dict(l=10, r=10, t=10, b=10), height=320)
            st.plotly_chart(fig, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<p class="footer-note">Tandoor · data: Zomato restaurant dataset</p>', unsafe_allow_html=True)


# ── Page 3: Predict a Rating ──────────────────────────────────────────
elif page == "⭐  Predict a Rating":
    st.markdown('<p class="kicker">Prediction</p>', unsafe_allow_html=True)
    st.title("Predict a restaurant's rating")
    st.caption("Describe a restaurant and a Random Forest model estimates the rating it's likely to earn.")

    left, right = st.columns(2)
    with left:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.subheader("Restaurant profile")
        countries = sorted(df["Country"].dropna().unique())
        country = st.selectbox("Country", countries, index=countries.index("India") if "India" in countries else 0)
        cost = st.number_input("Average cost for two", min_value=0, max_value=20000, value=1200, step=100)
        price_range = st.radio("Price range", [1, 2, 3, 4], index=2, horizontal=True)
        votes = st.number_input("Expected votes / reviews", min_value=0, max_value=20000, value=350, step=10)
        table_booking = st.radio("Table booking available?", ["No", "Yes"], index=1, horizontal=True)
        online_delivery = st.radio("Online delivery available?", ["No", "Yes"], index=1, horizontal=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.subheader("Predicted rating")

        country_enc = country_encoder.transform([country])[0] if country in country_encoder.classes_ else 0
        X_input = pd.DataFrame([{
            "Average Cost for two": cost, "Votes": votes, "Price range": price_range,
            "Delivery_enc": int(online_delivery == "Yes"), "Table_enc": int(table_booking == "Yes"),
            "Country_enc": country_enc,
        }])[feature_cols]
        predicted = float(np.clip(model.predict(X_input)[0], 0, 5))

        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number", value=predicted, number={"suffix": " / 5", "font": {"color": TEXT, "size": 34}},
            gauge={"axis": {"range": [0, 5], "tickcolor": MUTED}, "bar": {"color": ACCENT}, "bgcolor": PANEL,
                   "borderwidth": 0, "steps": [{"range": [0, 2.5], "color": "#3a2018"},
                                                {"range": [2.5, 3.5], "color": "#5a3420"},
                                                {"range": [3.5, 5], "color": "#7a4a28"}]},
        ))
        fig_gauge.update_layout(paper_bgcolor=PANEL, font_color=TEXT, height=260, margin=dict(l=20, r=20, t=30, b=10))
        st.plotly_chart(fig_gauge, use_container_width=True)

        label, color = rating_badge(predicted)
        st.markdown(f'<span class="badge" style="color:{color}; background:{color}22;">{label}</span>',
                     unsafe_allow_html=True)

        st.write("")
        st.markdown("**What's driving this**")
        name_map = {"Average Cost for two": "Avg cost for two", "Votes": "Votes", "Price range": "Price range",
                    "Delivery_enc": "Online delivery", "Table_enc": "Table booking", "Country_enc": "Country"}
        importances = pd.Series(model.feature_importances_, index=feature_cols)
        importances.index = [name_map[i] for i in importances.index]
        importances = importances.sort_values()

        fig_imp = go.Figure(go.Bar(x=importances.values, y=importances.index, orientation="h", marker_color=ACCENT))
        fig_imp.update_layout(template=PLOTLY_TEMPLATE, paper_bgcolor=PANEL, plot_bgcolor=PANEL, font_color=TEXT,
                               xaxis_title="Relative importance", margin=dict(l=10, r=10, t=10, b=10), height=220)
        st.plotly_chart(fig_imp, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<p class="footer-note">Model: Random Forest Regressor · trained live on the full dataset (cached)</p>',
                unsafe_allow_html=True)


# ── Page 4: Insights ───────────────────────────────────────────────────
else:
    top_country = df["Country"].value_counts().idxmax()
    top_country_pct = (df["Country"].value_counts(normalize=True).max() * 100).round(1)
    delivery_pct = (df["Has Online delivery"].value_counts(normalize=True).get("Yes", 0) * 100).round(1)
    top_feat = pd.Series(model.feature_importances_, index=feature_cols).idxmax()
    top_feat_name = {"Average Cost for two": "average cost for two", "Votes": "number of votes",
                      "Price range": "price range", "Delivery_enc": "online delivery",
                      "Table_enc": "table booking", "Country_enc": "country"}[top_feat]

    st.markdown('<p class="kicker">Insights</p>', unsafe_allow_html=True)
    st.title("Key Insights & Conclusion")
    st.caption("What the EDA, feature engineering, and prediction model actually taught us.")

    findings = [
        f"<b>{top_country}</b> accounts for the largest share of restaurants (~{top_country_pct}%), "
        "followed by the USA and UK.",
        f"Only about <b>{delivery_pct}%</b> of restaurants offer <b>online delivery</b>, and it is "
        "heavily concentrated in a few countries.",
        f"In the Random Forest model, <b>{top_feat_name}</b> is the single strongest predictor of a "
        "restaurant's rating — more influential than price or country alone.",
    ]

    left, right = st.columns([1.15, 1])
    with left:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.subheader("Findings from the EDA")
        for f in findings:
            st.markdown(f'<div style="border-left:3px solid {ACCENT}; padding:0.5rem 0.8rem; '
                        f'margin-bottom:0.6rem; background:{BG}; border-radius:4px; font-size:0.87rem;">'
                        f'📌 {f}</div>', unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.subheader("Feature engineering performed")
        st.markdown(
            "- Merged `zomato.csv` with `Country-Code.xlsx` for a readable `Country` column.\n"
            "- Converted table booking / online delivery into binary features.\n"
            "- Label-encoded `Country` into a numeric feature.\n"
            "- Filtered out un-rated restaurants before training."
        )
        st.markdown("</div>", unsafe_allow_html=True)

        st.write("")
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.subheader("Conclusion")
        st.markdown("A complete workflow: cleaning and joining data, visualizing patterns, "
                     "engineering features, and training a live, explainable prediction model.")
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<p class="footer-note">Tandoor · data: Zomato restaurant dataset · Thank you!</p>',
                unsafe_allow_html=True)
