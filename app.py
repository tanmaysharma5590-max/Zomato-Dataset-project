import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import os

# ----------------------------------------------------------------------------
# PAGE CONFIG
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="Tandoor | Zomato Restaurant Intelligence",
    page_icon="\U0001F3EA",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ----------------------------------------------------------------------------
# THEME — "Spice Market": charcoal-espresso base, turmeric + chili accents
# ----------------------------------------------------------------------------
CHARCOAL = "#1F1712"
CHARCOAL_2 = "#2A1F18"
PAPER = "#FBF4E8"
TURMERIC = "#E3A32D"
CHILI = "#C24C3B"
CARDAMOM = "#5F8B6E"
CLOVE = "#8C5A3C"
MUTED = "#B9A88F"

CUSTOM_CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:wght@600;700&family=Inter:wght@400;500;600&display=swap');

html, body, [class*="css"] {{
    font-family: 'Inter', sans-serif;
}}

.stApp {{
    background-color: {CHARCOAL};
    color: {PAPER};
}}

section[data-testid="stSidebar"] {{
    background-color: {CHARCOAL_2};
    border-right: 1px solid #3A2C21;
}}

section[data-testid="stSidebar"] * {{
    color: {PAPER} !important;
}}

h1, h2, h3 {{
    font-family: 'Fraunces', serif !important;
    color: {PAPER} !important;
    letter-spacing: -0.01em;
}}

.tandoor-eyebrow {{
    color: {TURMERIC};
    text-transform: uppercase;
    letter-spacing: 0.18em;
    font-size: 0.75rem;
    font-weight: 600;
    margin-bottom: 0.3rem;
}}

.tandoor-hero {{
    border-bottom: 1px solid #3A2C21;
    padding-bottom: 1.6rem;
    margin-bottom: 1.8rem;
}}

.metric-card {{
    background: linear-gradient(180deg, {CHARCOAL_2} 0%, #241A13 100%);
    border: 1px solid #3A2C21;
    border-left: 3px solid {TURMERIC};
    border-radius: 6px;
    padding: 1.1rem 1.3rem;
}}

.metric-card .label {{
    color: {MUTED};
    font-size: 0.78rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}}

.metric-card .value {{
    font-family: 'Fraunces', serif;
    font-size: 2.1rem;
    color: {PAPER};
    line-height: 1.1;
    margin-top: 0.15rem;
}}

.chili-card {{
    border-left: 3px solid {CHILI} !important;
}}
.cardamom-card {{
    border-left: 3px solid {CARDAMOM} !important;
}}
.clove-card {{
    border-left: 3px solid {CLOVE} !important;
}}

div[data-testid="stMetricValue"] {{
    color: {PAPER};
}}

.stButton>button {{
    background-color: {TURMERIC};
    color: {CHARCOAL};
    border: none;
    border-radius: 4px;
    font-weight: 600;
    padding: 0.55rem 1.4rem;
    letter-spacing: 0.02em;
}}
.stButton>button:hover {{
    background-color: #F0B747;
    color: {CHARCOAL};
}}

hr {{
    border-color: #3A2C21;
}}

.dataframe {{
    background-color: {CHARCOAL_2};
}}

.footer-note {{
    color: {MUTED};
    font-size: 0.8rem;
    margin-top: 3rem;
    border-top: 1px solid #3A2C21;
    padding-top: 1rem;
}}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

PLOTLY_TEMPLATE = go.layout.Template()
PLOTLY_TEMPLATE.layout = go.Layout(
    paper_bgcolor=CHARCOAL_2,
    plot_bgcolor=CHARCOAL_2,
    font=dict(color=PAPER, family="Inter"),
    colorway=[TURMERIC, CHILI, CARDAMOM, CLOVE, MUTED, "#7A8FAE"],
    xaxis=dict(gridcolor="#3A2C21", zerolinecolor="#3A2C21"),
    yaxis=dict(gridcolor="#3A2C21", zerolinecolor="#3A2C21"),
    legend=dict(bgcolor="rgba(0,0,0,0)"),
)

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

# ----------------------------------------------------------------------------
# DATA LOADING
# ----------------------------------------------------------------------------
@st.cache_data
def load_data():
    csv_path = os.path.join(DATA_DIR, "zomato.csv")
    xlsx_path = os.path.join(DATA_DIR, "Country-Code.xlsx")

    missing = [p for p in [csv_path, xlsx_path] if not os.path.exists(p)]
    if missing:
        st.error(
            "**Data files not found.** This app expects a `data/` folder sitting right next to "
            "`app.py`, containing `zomato.csv` and `Country-Code.xlsx`.\n\n"
            f"Missing: {', '.join(os.path.basename(p) for p in missing)}\n\n"
            "If you're on **Streamlit Cloud**: this usually means the `data/` folder wasn't pushed "
            "to your GitHub repo (folders with only large files are sometimes skipped, or `.gitignore` "
            "is excluding them). Check that your repo has this structure:\n\n"
            "```\n"
            "your-repo/\n"
            "├── app.py\n"
            "├── requirements.txt\n"
            "└── data/\n"
            "    ├── zomato.csv\n"
            "    └── Country-Code.xlsx\n"
            "```\n\n"
            "If they're missing from GitHub, add them (`git add data/`, commit, push) and reboot the app "
            "from **Manage app → Reboot**."
        )
        st.stop()

    df = pd.read_csv(csv_path, encoding="latin-1")
    cc = pd.read_excel(xlsx_path)
    df = df.merge(cc, on="Country Code", how="left")
    df["Cuisines"] = df["Cuisines"].fillna("Not Specified")
    df["Primary Cuisine"] = df["Cuisines"].apply(lambda x: str(x).split(",")[0].strip())
    df = df[df["Aggregate rating"] >= 0]
    return df


@st.cache_resource
def train_model(df: pd.DataFrame):
    features = df[[
        "Average Cost for two", "Price range", "Votes",
        "Has Table booking", "Has Online delivery", "Country Code",
    ]].copy()

    le_table = LabelEncoder()
    le_online = LabelEncoder()
    features["Has Table booking"] = le_table.fit_transform(features["Has Table booking"])
    features["Has Online delivery"] = le_online.fit_transform(features["Has Online delivery"])

    target = df["Aggregate rating"]

    model = RandomForestRegressor(n_estimators=200, max_depth=12, random_state=42, n_jobs=-1)
    model.fit(features, target)

    return model, le_table, le_online, features.columns.tolist()


df = load_data()
model, le_table, le_online, feature_cols = train_model(df)

# ----------------------------------------------------------------------------
# SIDEBAR NAV
# ----------------------------------------------------------------------------
st.sidebar.markdown("### \U0001F3EA Tandoor")
st.sidebar.caption("Zomato Restaurant Intelligence")
page = st.sidebar.radio(
    "Navigate",
    ["\U0001F3E0  Overview", "\U0001F50D  Explore the Data", "\u2B50  Predict a Rating"],
    label_visibility="collapsed",
)
st.sidebar.markdown("---")
st.sidebar.caption(f"Dataset: {len(df):,} restaurants across {df['Country'].nunique()} countries")
st.sidebar.caption("Source: Zomato restaurant dataset")

# ============================================================================
# PAGE 1 — OVERVIEW
# ============================================================================
if page.startswith("\U0001F3E0"):
    st.markdown('<div class="tandoor-hero">', unsafe_allow_html=True)
    st.markdown('<div class="tandoor-eyebrow">Restaurant Intelligence</div>', unsafe_allow_html=True)
    st.markdown("# What makes a restaurant well-rated?")
    st.write(
        "An exploration of **{:,} restaurants** across **{} countries**, drawn from the Zomato "
        "dataset — built to understand what drives customer ratings, and to predict a rating "
        "before a restaurant even opens its doors.".format(len(df), df["Country"].nunique())
    )
    st.markdown('</div>', unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""<div class="metric-card"><div class="label">Restaurants</div>
        <div class="value">{len(df):,}</div></div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""<div class="metric-card chili-card"><div class="label">Avg. Rating</div>
        <div class="value">{df['Aggregate rating'].mean():.2f} / 5</div></div>""", unsafe_allow_html=True)
    with c3:
        st.markdown(f"""<div class="metric-card cardamom-card"><div class="label">Cities Covered</div>
        <div class="value">{df['City'].nunique()}</div></div>""", unsafe_allow_html=True)
    with c4:
        st.markdown(f"""<div class="metric-card clove-card"><div class="label">Cuisines Tracked</div>
        <div class="value">{df['Primary Cuisine'].nunique()}</div></div>""", unsafe_allow_html=True)

    st.write("")
    st.write("")

    colA, colB = st.columns([1.3, 1])
    with colA:
        st.markdown("#### Rating distribution")
        fig = px.histogram(
            df[df["Aggregate rating"] > 0], x="Aggregate rating", nbins=30,
            template=PLOTLY_TEMPLATE, color_discrete_sequence=[TURMERIC],
        )
        fig.update_layout(height=340, margin=dict(t=10, l=10, r=10, b=10), bargap=0.05)
        st.plotly_chart(fig, use_container_width=True)

    with colB:
        st.markdown("#### Top 8 cuisines")
        top_cuisine = df["Primary Cuisine"].value_counts().head(8).sort_values()
        fig2 = px.bar(
            top_cuisine, x=top_cuisine.values, y=top_cuisine.index, orientation="h",
            template=PLOTLY_TEMPLATE, color_discrete_sequence=[CHILI],
        )
        fig2.update_layout(height=340, margin=dict(t=10, l=10, r=10, b=10),
                            xaxis_title="Restaurants", yaxis_title="")
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("#### Top 10 cities by restaurant count")
    top_cities = df["City"].value_counts().head(10).sort_values()
    fig3 = px.bar(
        top_cities, x=top_cities.values, y=top_cities.index, orientation="h",
        template=PLOTLY_TEMPLATE, color_discrete_sequence=[CARDAMOM],
    )
    fig3.update_layout(height=380, margin=dict(t=10, l=10, r=10, b=10),
                        xaxis_title="Restaurants", yaxis_title="")
    st.plotly_chart(fig3, use_container_width=True)

    st.markdown(
        '<div class="footer-note">Tandoor · built with Streamlit · data: Zomato restaurant dataset</div>',
        unsafe_allow_html=True,
    )

# ============================================================================
# PAGE 2 — EXPLORE (EDA)
# ============================================================================
elif page.startswith("\U0001F50D"):
    st.markdown('<div class="tandoor-eyebrow">Exploration</div>', unsafe_allow_html=True)
    st.markdown("# Explore the data")
    st.write("Filter the dataset and see how price, votes, delivery and booking options move the rating.")

    f1, f2, f3 = st.columns(3)
    with f1:
        countries = ["All"] + sorted(df["Country"].dropna().unique().tolist())
        sel_country = st.selectbox("Country", countries)
    with f2:
        price_opts = ["All"] + sorted(df["Price range"].unique().tolist())
        sel_price = st.selectbox("Price range (1=cheap, 4=premium)", price_opts)
    with f3:
        cuisines = ["All"] + sorted(df["Primary Cuisine"].value_counts().head(30).index.tolist())
        sel_cuisine = st.selectbox("Primary cuisine (top 30)", cuisines)

    fdf = df.copy()
    if sel_country != "All":
        fdf = fdf[fdf["Country"] == sel_country]
    if sel_price != "All":
        fdf = fdf[fdf["Price range"] == sel_price]
    if sel_cuisine != "All":
        fdf = fdf[fdf["Primary Cuisine"] == sel_cuisine]

    st.caption(f"Showing **{len(fdf):,}** restaurants matching the filters")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### Rating by price range")
        fig = px.box(
            fdf[fdf["Aggregate rating"] > 0], x="Price range", y="Aggregate rating",
            template=PLOTLY_TEMPLATE, color="Price range",
            color_discrete_sequence=[TURMERIC, CHILI, CARDAMOM, CLOVE],
        )
        fig.update_layout(height=360, margin=dict(t=10, l=10, r=10, b=10), showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
        st.caption(
            "📖 **In plain terms:** each box shows the typical rating range for restaurants at that "
            "price level (1 = cheap, 4 = premium). A higher box means better ratings on average."
        )

    with c2:
        st.markdown("#### Online delivery vs. rating")
        grp = fdf.groupby("Has Online delivery")["Aggregate rating"].mean().reset_index()
        fig2 = px.bar(
            grp, x="Has Online delivery", y="Aggregate rating",
            template=PLOTLY_TEMPLATE, color="Has Online delivery",
            color_discrete_sequence=[MUTED, CARDAMOM],
        )
        fig2.update_layout(height=360, margin=dict(t=10, l=10, r=10, b=10), showlegend=False)
        st.plotly_chart(fig2, use_container_width=True)
        st.caption(
            "📖 **In plain terms:** the two bars compare average rating for restaurants without vs. "
            "with online delivery. The taller bar rates higher."
        )

    st.markdown("#### Votes vs. rating (bubble = cost for two)")
    plot_df = fdf[(fdf["Aggregate rating"] > 0) & (fdf["Votes"] > 0)].sample(
        min(1500, len(fdf[(fdf["Aggregate rating"] > 0) & (fdf["Votes"] > 0)])), random_state=1
    ) if len(fdf) > 0 else fdf
    if len(plot_df) > 0:
        fig3 = px.scatter(
            plot_df, x="Votes", y="Aggregate rating", size="Average Cost for two",
            color="Price range", template=PLOTLY_TEMPLATE, opacity=0.7,
            color_continuous_scale=[TURMERIC, CHILI],
            log_x=True,
        )
        fig3.update_layout(height=420, margin=dict(t=10, l=10, r=10, b=10))
        st.plotly_chart(fig3, use_container_width=True)
        st.caption(
            "📖 **In plain terms:** each dot is a restaurant. Further right = more reviews. Higher up = "
            "better rating. Bigger, redder dots = pricier restaurants. Most restaurants cluster between "
            "3 and 4 stars regardless of how many reviews they have."
        )
    else:
        st.info("No rows match this filter combination.")

    st.markdown("#### Table booking vs. rating")
    grp2 = fdf.groupby("Has Table booking")["Aggregate rating"].mean().reset_index()
    fig4 = px.bar(
        grp2, x="Has Table booking", y="Aggregate rating",
        template=PLOTLY_TEMPLATE, color="Has Table booking",
        color_discrete_sequence=[MUTED, CLOVE],
    )
    fig4.update_layout(height=320, margin=dict(t=10, l=10, r=10, b=10), showlegend=False)
    st.plotly_chart(fig4, use_container_width=True)
    st.caption(
        "📖 **In plain terms:** restaurants that let you reserve a table in advance (right bar) tend "
        "to rate higher than those that don't (left bar)."
    )

    with st.expander("View filtered raw data"):
        st.dataframe(
            fdf[["Restaurant Name", "City", "Country", "Cuisines", "Average Cost for two",
                 "Price range", "Votes", "Aggregate rating"]].reset_index(drop=True),
            use_container_width=True, height=300,
        )

# ============================================================================
# PAGE 3 — PREDICT
# ============================================================================
else:
    st.markdown('<div class="tandoor-eyebrow">Prediction</div>', unsafe_allow_html=True)
    st.markdown("# Predict a restaurant's rating")
    st.write(
        "Describe a restaurant and a Random Forest model — trained on the full Zomato dataset — "
        "estimates the aggregate rating it's likely to earn."
    )

    left, right = st.columns([1, 1.2])

    with left:
        st.markdown("#### Restaurant profile")
        country_name = st.selectbox("Country", sorted(df["Country"].dropna().unique().tolist()),
                                     index=sorted(df["Country"].dropna().unique().tolist()).index("India")
                                     if "India" in df["Country"].values else 0)
        country_code = df[df["Country"] == country_name]["Country Code"].iloc[0]

        cost = st.slider("Average cost for two", min_value=50, max_value=8000, value=800, step=50)
        price_range = st.select_slider("Price range", options=[1, 2, 3, 4], value=2)
        votes = st.slider("Expected votes / reviews", min_value=0, max_value=2000, value=100, step=10)
        table_booking = st.radio("Table booking available?", ["No", "Yes"], horizontal=True)
        online_delivery = st.radio("Online delivery available?", ["No", "Yes"], horizontal=True)

        predict_clicked = st.button("Predict rating", use_container_width=True)

    with right:
        st.markdown("#### Predicted rating")
        if predict_clicked:
            row = pd.DataFrame([{
                "Average Cost for two": cost,
                "Price range": price_range,
                "Votes": votes,
                "Has Table booking": le_table.transform([table_booking])[0],
                "Has Online delivery": le_online.transform([online_delivery])[0],
                "Country Code": country_code,
            }])[feature_cols]

            pred = float(model.predict(row)[0])
            pred = max(0, min(5, pred))

            gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=pred,
                number={"suffix": " / 5", "font": {"size": 44, "color": PAPER}},
                gauge={
                    "axis": {"range": [0, 5], "tickcolor": PAPER},
                    "bar": {"color": TURMERIC},
                    "bgcolor": CHARCOAL_2,
                    "steps": [
                        {"range": [0, 2.5], "color": "#3A2C21"},
                        {"range": [2.5, 3.5], "color": "#4A3826"},
                        {"range": [3.5, 5], "color": "#5A4530"},
                    ],
                },
            ))
            gauge.update_layout(
                paper_bgcolor=CHARCOAL, font={"color": PAPER}, height=320,
                margin=dict(t=30, l=30, r=30, b=10),
            )
            st.plotly_chart(gauge, use_container_width=True)

            if pred >= 4:
                st.success("Excellent — this profile matches highly-rated restaurants in the dataset.")
            elif pred >= 3:
                st.info("Good — a solid, well-regarded rating range.")
            elif pred >= 2:
                st.warning("Average — there's room to strengthen the offering.")
            else:
                st.error("Below average — this combination tends to rate poorly in the data.")

            st.markdown("##### What's driving this")
            importances = pd.Series(model.feature_importances_, index=feature_cols).sort_values()
            fig_imp = px.bar(
                importances, x=importances.values, y=importances.index, orientation="h",
                template=PLOTLY_TEMPLATE, color_discrete_sequence=[CHILI],
            )
            fig_imp.update_layout(height=260, margin=dict(t=10, l=10, r=10, b=10),
                                   xaxis_title="Relative importance", yaxis_title="")
            st.plotly_chart(fig_imp, use_container_width=True)
        else:
            st.info("Set the restaurant profile on the left, then click **Predict rating**.")

    st.markdown(
        '<div class="footer-note">Model: Random Forest Regressor · trained live on the full dataset '
        'each session (cached) · for demonstration purposes</div>',
        unsafe_allow_html=True,
    )
