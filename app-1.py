"""
Zomato EDA Streamlit App
Three pages: Home, EDA Dashboard, Data Explorer & Insights.

Run with:
    pip install -r requirements.txt
    streamlit run app.py
"""

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(
    page_title="Zomato EDA Streamlit App",
    page_icon="🍴",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Dark theme ─────────────────────────────────────────────────────────
BG = "#141414"
PANEL = "#1E1E1E"
BORDER = "#333333"
TEXT = "#F2F2F2"
MUTED = "#A0A0A0"
RED = "#C0392B"
RED_DARK = "#8E2A20"
GOLD = "#E0A73C"
GREEN = "#3FA66B"
GREY = "#6E6E6E"
PLOTLY_TEMPLATE = "plotly_dark"

st.markdown(f"""
<style>
.stApp {{ background-color: {BG}; color: {TEXT}; }}
section[data-testid="stSidebar"] {{ background-color: #0F0F0F; border-right: 1px solid {BORDER}; }}
h1, h2, h3, h4, p, span, label {{ color: {TEXT}; }}
.brand-title {{ color: {RED}; font-size: 1.2rem; font-weight: 800; margin-bottom: 0; }}
.brand-sub {{ color: {MUTED}; font-size: 0.72rem; text-transform: uppercase; margin-top: -4px; }}
.sidebar-footer {{ color: {MUTED}; font-size: 0.72rem; border-top: 1px solid {BORDER}; padding-top: 0.7rem; margin-top: 1rem; }}
.header-band {{ background: linear-gradient(90deg, {RED} 0%, {RED_DARK} 100%); border-radius: 10px;
                padding: 1rem 1.3rem; margin-bottom: 1rem; }}
.header-band h2 {{ color: white; margin: 0; }}
.header-band p {{ color: #f4d9d5; margin: 0.2rem 0 0 0; font-size: 0.85rem; }}
.panel {{ background: {PANEL}; border: 1px solid {BORDER}; border-radius: 12px; padding: 1.1rem 1.3rem; }}
.metric-card {{ background: {PANEL}; border: 1px solid {BORDER}; border-radius: 10px; padding: 0.9rem 1rem; text-align: center; }}
.metric-card .value {{ font-size: 1.5rem; font-weight: 800; color: {RED}; }}
.metric-card .label {{ font-size: 0.75rem; color: {MUTED}; margin-top: 2px; }}
.insight-item {{ border-left: 3px solid {RED}; padding: 0.5rem 0.8rem; margin-bottom: 0.6rem;
                  background: {BG}; border-radius: 4px; font-size: 0.87rem; }}
.footer-note {{ text-align: center; color: {MUTED}; font-size: 0.75rem; margin-top: 2rem; }}
</style>
""", unsafe_allow_html=True)


# ── Data ───────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv("data/zomato.csv", encoding="latin-1")
    country = pd.read_excel("data/Country-Code.xlsx")
    df = pd.merge(df, country, on="Country Code", how="left")
    return df


df = load_data()
rated = df[df["Aggregate rating"] > 0]

# ── Sidebar ────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<p class="brand-title">🍴 Zomato Project</p>', unsafe_allow_html=True)
    st.markdown('<p class="brand-sub">EDA Streamlit App</p>', unsafe_allow_html=True)
    st.write("")
    page = st.radio("Navigate", ["Home", "EDA Dashboard", "Data Explorer & Insights"],
                     label_visibility="collapsed")
    st.markdown(
        f'<div class="sidebar-footer">Dataset: {df.shape[0]:,} restaurants across '
        f'{df["Country"].nunique()} countries<br>Source: Zomato restaurant dataset</div>',
        unsafe_allow_html=True,
    )


# ── Page 1: Home ─────────────────────────────────────────────────────
if page == "Home":
    st.markdown(
        '<div class="header-band"><h2>🍴 Zomato Dataset — EDA & Feature Engineering</h2>'
        '<p>Interactive dashboard built with Python + Streamlit, based on the EDA & feature engineering notebook.</p></div>',
        unsafe_allow_html=True)

    stats = [
        (f"{df.shape[0]:,}", "Restaurants"),
        (f"{df['Country'].nunique()}", "Countries"),
        (f"{df['City'].nunique()}", "Cities"),
        (f"{df['Cuisines'].nunique():,}", "Cuisine Combos"),
        (f"{rated['Aggregate rating'].mean():.2f} ★", "Avg. Rating"),
    ]
    cols = st.columns(5)
    for col, (val, lab) in zip(cols, stats):
        col.markdown(f'<div class="metric-card"><div class="value">{val}</div>'
                      f'<div class="label">{lab}</div></div>', unsafe_allow_html=True)

    st.write("")
    left, right = st.columns([1.3, 1])
    with left:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.subheader("What this project does")
        st.markdown(
            "This project takes Zomato's public restaurant dataset "
            f"({df.shape[0]:,}+ restaurants worldwide) through a full EDA and Feature "
            "Engineering pipeline:\n\n"
            "1. Load & clean the raw data, merge with the country-code lookup table.\n"
            "2. Check data quality — missing values, data types, duplicates.\n"
            "3. Explore patterns — country usage, ratings, cost, delivery.\n"
            "4. Engineer features — rating buckets, encoded categorical columns.\n"
            "5. Present findings in this interactive dashboard for any audience."
        )
        st.markdown("</div>", unsafe_allow_html=True)
    with right:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.subheader("No coding needed")
        st.markdown(
            "Click through pages in the sidebar. Every chart is interactive — "
            "hover and filter with dropdowns."
        )
        st.markdown("</div>", unsafe_allow_html=True)
        st.write("")
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.subheader("For the technical crowd")
        st.markdown("Full source in `app.py` and `requirements.txt`. Plain pandas + Plotly, no magic.")
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<p class="footer-note">Zomato EDA Streamlit App</p>', unsafe_allow_html=True)


# ── Page 2: EDA Dashboard ─────────────────────────────────────────────
elif page == "EDA Dashboard":
    st.markdown(
        '<div class="header-band"><h2>📊 Exploratory Data Analysis</h2>'
        '<p>Interactive charts — filter by country using the sidebar.</p></div>',
        unsafe_allow_html=True)

    country_filter = st.multiselect("Filter by country (optional)", sorted(df["Country"].dropna().unique()))
    view = df[df["Country"].isin(country_filter)] if country_filter else df
    view_rated = view[view["Aggregate rating"] > 0]

    tab_countries, tab_ratings, tab_delivery, tab_price, tab_cuisines = st.tabs(
        ["Countries", "Ratings", "Delivery", "Price Range", "Cuisines"])

    with tab_countries:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown("**Top countries by restaurant count**")
        top_c = view["Country"].value_counts().head(8)
        fig = px.pie(values=top_c.values, names=top_c.index, hole=0.45,
                     color_discrete_sequence=px.colors.sequential.Reds_r)
        fig.update_layout(template=PLOTLY_TEMPLATE, paper_bgcolor=PANEL, font_color=TEXT,
                           margin=dict(l=10, r=10, t=10, b=10), height=380)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with tab_ratings:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown("**Rating categories**")
        cat_order = ["Excellent", "Very Good", "Good", "Average", "Poor", "Not rated"]
        cat_colors = {"Excellent": "#2E8B57", "Very Good": GREEN, "Good": GOLD,
                      "Average": "#C9862E", "Poor": RED, "Not rated": GREY}
        counts = view["Rating text"].value_counts().reindex(cat_order).dropna()
        fig = go.Figure(go.Bar(x=counts.index, y=counts.values,
                                marker_color=[cat_colors[c] for c in counts.index]))
        fig.update_layout(template=PLOTLY_TEMPLATE, paper_bgcolor=PANEL, plot_bgcolor=PANEL,
                           font_color=TEXT, yaxis_title="Restaurants",
                           margin=dict(l=10, r=10, t=10, b=10), height=380)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with tab_delivery:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown("**Online delivery by country** (top 10 by restaurant count)")
        top10 = view["Country"].value_counts().head(10).index
        delivery = (view[view["Country"].isin(top10)]
                    .groupby("Country")["Has Online delivery"]
                    .apply(lambda s: (s == "Yes").mean() * 100)
                    .sort_values())
        fig = go.Figure(go.Bar(x=delivery.values, y=delivery.index, orientation="h",
                                marker_color=RED))
        fig.update_layout(template=PLOTLY_TEMPLATE, paper_bgcolor=PANEL, plot_bgcolor=PANEL,
                           font_color=TEXT, xaxis_title="% offering online delivery",
                           margin=dict(l=10, r=10, t=10, b=10), height=380)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with tab_price:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown("**Cost for two by price range**")
        fig = px.box(view[view["Average Cost for two"] > 0], x="Price range", y="Average Cost for two",
                      color="Price range", color_discrete_sequence=px.colors.sequential.Reds)
        fig.update_layout(template=PLOTLY_TEMPLATE, paper_bgcolor=PANEL, plot_bgcolor=PANEL,
                           font_color=TEXT, showlegend=False,
                           margin=dict(l=10, r=10, t=10, b=10), height=380)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with tab_cuisines:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown("**Most common cuisines**")
        cuisines = view["Cuisines"].dropna().str.split(", ").explode().value_counts().head(12).sort_values()
        fig = go.Figure(go.Bar(x=cuisines.values, y=cuisines.index, orientation="h", marker_color=GOLD))
        fig.update_layout(template=PLOTLY_TEMPLATE, paper_bgcolor=PANEL, plot_bgcolor=PANEL,
                           font_color=TEXT, xaxis_title="Restaurants",
                           margin=dict(l=10, r=10, t=10, b=10), height=420)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<p class="footer-note">Zomato EDA Streamlit App — data: Zomato restaurant dataset</p>',
                unsafe_allow_html=True)


# ── Page 3: Data Explorer & Insights ──────────────────────────────────
else:
    st.markdown(
        '<div class="header-band"><h2>🔎 Data Explorer / 💡 Insights</h2>'
        '<p>Search & filter restaurants, then review key findings — no code required.</p></div>',
        unsafe_allow_html=True)

    left, right = st.columns([1.4, 1])

    with left:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.subheader("Search & Filter")
        c1, c2, c3 = st.columns(3)
        name_q = c1.text_input("Restaurant name contains")
        city_q = c2.selectbox("City", ["All"] + sorted(df["City"].dropna().unique().tolist()))
        min_rating = c3.slider("Min rating (0.0)", 0.0, 5.0, 0.0, 0.1)

        result = df.copy()
        if name_q:
            result = result[result["Restaurant Name"].str.contains(name_q, case=False, na=False)]
        if city_q != "All":
            result = result[result["City"] == city_q]
        result = result[result["Aggregate rating"] >= min_rating]

        st.dataframe(
            result[["Restaurant Name", "City", "Country", "Cuisines",
                    "Average Cost for two", "Aggregate rating", "Votes"]].head(200),
            use_container_width=True, hide_index=True, height=380)
        st.caption(f"{len(result):,} matching restaurants (showing up to 200)")
        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.subheader("Key Insights")
        top_country = df["Country"].value_counts().idxmax()
        top_country_pct = (df["Country"].value_counts(normalize=True).max() * 100).round(1)
        second_third = df["Country"].value_counts().index[1:3].tolist()
        delivery_pct = (df["Has Online delivery"].value_counts(normalize=True).get("Yes", 0) * 100).round(1)
        mid_share = (rated["Aggregate rating"].between(2.5, 3.4).mean() * 100).round(1)
        cost_corr = rated[["Average Cost for two", "Aggregate rating"]].corr().iloc[0, 1]

        insights = [
            f"<b>{top_country}</b> accounts for the largest share of restaurants "
            f"(~{top_country_pct}%), followed by {second_third[0]} and {second_third[1]}.",
            f"Most rated restaurants cluster in the 2.5–3.4 average rating band "
            f"(~{mid_share}% of rated restaurants).",
            f"Only about <b>{delivery_pct}%</b> of restaurants offer online delivery.",
            f"Average cost for two rises {'predictably' if cost_corr > 0 else 'inconsistently'} "
            f"with the restaurant's price range.",
        ]
        for ins in insights:
            st.markdown(f'<div class="insight-item">📌 {ins}</div>', unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<p class="footer-note">Zomato EDA Streamlit App — data: Zomato restaurant dataset · Thank you!</p>',
                unsafe_allow_html=True)
