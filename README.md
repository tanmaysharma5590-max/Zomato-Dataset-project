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

## Streamlit Cloud par deploy karte waqt

GitHub repo mein zaroor ye structure hona chahiye:

```
your-repo/
├── app.py
├── requirements.txt
└── data/
    ├── zomato.csv
    └── Country-Code.xlsx
```

`FileNotFoundError` aaye to iska matlab `data/` folder GitHub repo mein push nahi hua hai (`.gitignore` check karo, kabhi kabhi CSV files accidentally ignore ho jaati hain). Fix:

```bash
git add data/
git commit -m "add data files"
git push
```

Phir Streamlit Cloud dashboard mein **Manage app → Reboot** karo.
