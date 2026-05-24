# CarWise Fuzzy — Used-Car Fuzzy DSS

CarWise Fuzzy is a final implementation prototype for a fuzzy decision support system for used-car purchase evaluation.

## What the app does

The app asks the buyer simple human-centered questions, such as:

- priority: Budget, Balanced, Reliability
- main use: city commute, family use, long-distance, student budget, weekend trips
- driving context: city, highway, mixed, hilly/snowy areas
- preferred fuel and transmission
- maximum comfortable budget
- reliability expectation

The app then ranks cars from the 50-car evaluation dataset using the original seven fuzzy input dimensions:

1. price
2. mileage
3. age
4. condition
5. service history
6. accident risk
7. seller trust

Extra buyer questions do not replace the original model. They only adjust the weighting/filtering of the seven fuzzy dimensions.

## Files

- `app.py` — Streamlit web app
- `usedcar_fuzzy_evaluation_dataset_50.csv` — dataset with 30 real AutoScout24 samples and 20 synthetic balancing cases
- `requirements.txt` — Python dependencies
- `README.md` — this file

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy on Streamlit Community Cloud

1. Create a GitHub repository.
2. Upload `app.py`, `requirements.txt`, `README.md`, and `usedcar_fuzzy_evaluation_dataset_50.csv`.
3. Go to Streamlit Community Cloud.
4. Create a new app from your GitHub repository.
5. Set the app entry file to `app.py`.
6. Deploy.

## Prototype note

The current dataset is mainly sedan-focused, because the real AutoScout24 samples were collected from sedan listings. The app includes body-type preference as a soft user preference and clearly avoids pretending that unsupported car categories are fully covered. Future work can expand the dataset with SUVs, hatchbacks, and real vehicle images.
