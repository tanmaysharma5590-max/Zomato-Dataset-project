# Tandoor — Zomato Restaurant Intelligence (Streamlit App)

3-page Streamlit app, Zomato dataset (`zomato.csv`) par based:

1. **Overview** — key metrics, rating distribution, top cuisines, top cities
2. **Explore the Data** — filters (country / price / cuisine) + interactive charts
3. **Predict a Rating** — Random Forest model se restaurant profile daal ke rating predict karo

## Run karne ke liye

```bash
pip install -r requirements.txt
streamlit run app.py
```

Data files (`data/zomato.csv`, `data/Country-Code.xlsx`) already isi folder ke andar hain — kahin aur move mat karna, warna app data load nahi kar payega.

App browser me `http://localhost:8501` par khulega.
