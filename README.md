# CarWise Fuzzy — Final Streamlit Prototype (55 real cars)

This updated version uses a 55-car real AutoScout24 dataset covering sedans, SUVs, pickup trucks, hybrid vehicles, and electric vehicles.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Images

Optional real car images can be added in an `images/` folder using names that match the car sequence, for example:

```text
images/1.webp
images/2.webp
images/8.jpg
```

The app automatically falls back to a generated placeholder if an image is missing.

## Deployment

Upload these files to GitHub and Streamlit Cloud will redeploy the app:

- app.py
- requirements.txt
- carwise_fuzzy_dataset_55_real.csv
- images/ folder, optional
