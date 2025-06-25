from fastapi import FastAPI
from fastapi.responses import JSONResponse
import pandas as pd
from sklearn.linear_model import LinearRegression
from datetime import datetime, timedelta
import uvicorn
from fastapi.middleware.cors import CORSMiddleware
 
app = FastAPI()
 
 
# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure this for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
 
@app.get("/forecast")
def get_forecast():
    # 1. Load the CSV
    df = pd.read_csv("daily_climate.csv")
 
    # 2. Parse and combine year/month/day into datetime
    df.rename(columns={'YEAR': 'year', 'MO': 'month', 'DY': 'day'}, inplace=True)
    df['Date'] = pd.to_datetime(df[['year', 'month', 'day']])
    df.set_index('Date', inplace=True)
 
    # 3. Keep only the relevant columns
    variables = ['Temprature', 'Humidity', 'Rainfall', 'Wind']
    df = df[variables]
 
    # 4. Create lag features
    for var in variables:
        df[f'{var}_t-1'] = df[var].shift(1)
        df[f'{var}_t-2'] = df[var].shift(2)
 
    df.dropna(inplace=True)
 
    # 5. Train a model per variable
    models = {}
    for var in variables:
        X = df[[f'{var}_t-1', f'{var}_t-2']]
        y = df[var]
        model = LinearRegression()
        model.fit(X, y)
        models[var] = model
 
    # 6. Get last two known values for prediction
    latest = df[-2:].copy()
    last_inputs = {}
    for var in variables:
        last_inputs[var] = [
            latest[var].iloc[-1],
            latest[var].iloc[-2]
        ]
 
    # 7. Forecast next 7 days from today
    start_date = datetime.today().date()
    forecast_dates = [start_date + timedelta(days=i) for i in range(1, 8)]
    forecast_data = {var: [] for var in variables}
 
    for _ in range(7):
        for var in variables:
            input_df = pd.DataFrame([last_inputs[var]], columns=[f'{var}_t-1', f'{var}_t-2'])
            prediction = models[var].predict(input_df)[0]
            forecast_data[var].append(prediction)
 
            last_inputs[var] = [prediction, last_inputs[var][0]]
 
    # 8. Create forecast DataFrame
    forecast_df = pd.DataFrame(forecast_data, index=pd.to_datetime(forecast_dates))
    forecast_df.columns = [f'{col}_Forecast' for col in forecast_df.columns]
 
    # Convert Timestamp index to ISO format string
    forecast_df.reset_index(inplace=True)
    forecast_df.rename(columns={"index": "Date"}, inplace=True)
    forecast_df["Date"] = forecast_df["Date"].dt.strftime("%Y-%m-%d")
 
    # 9. Return as JSON
    return JSONResponse(content=forecast_df.to_dict(orient="records"))
 
# Add a main function for running locally
def main():
    # Use uvicorn to run the app when this script is executed
    uvicorn.run(app, host="localhost", port=1000, reload=True)
 
# Check if the script is run directly and not imported
if __name__ == "__main__":
    main()
 
 