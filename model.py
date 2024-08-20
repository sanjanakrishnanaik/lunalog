from datetime import datetime
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from datetime import datetime, timedelta

app = FastAPI()

# Define the request model
class PeriodRequest(BaseModel):
    dateIp: str

@app.post("/")
def get_period(request: PeriodRequest):
    try:
        # Parse the input date
        date = datetime.strptime(request.dateIp, "%Y-%m-%d").date()
        current_date = datetime.now().date()

        # Calculate days since period start
        days_since_period_start = (current_date - date).days

        # Define symptom probabilities based on days since period start
        symptom_probabilities = {
            'cramps': 0.5 if days_since_period_start <= 3 else 0.2,
            'mood_swings': 0.7 if days_since_period_start <= 5 else 0.3,
            'fatigue': 0.8 if days_since_period_start <= 7 else 0.4
        }

        # Simulate symptoms based on probabilities
        symptoms = {symptom: 1 if np.random.rand() < probability else 0
                    for symptom, probability in symptom_probabilities.items()}

        return symptoms
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid date format. Please use yyyy-mm-dd.")
    
    
    
    
    
    
    
    
class NextRequest(BaseModel):
    cycle_length: int
    last_period_date: str

@app.post("/next")
def get_dates(request: NextRequest):
    try:
        data = {
            'cycle_length': [
                28, 30, 27, 29, 31, 26, 32, 28, 29, 27,
                30, 28, 26, 32, 31, 30, 27, 29, 28, 30,
                31, 32, 27, 26, 29, 28, 30, 31, 27, 32,
                28, 30, 27, 29, 31, 26, 32, 28, 29, 27,
                30, 28, 26, 32, 31, 30, 27, 29, 28, 30,
                31, 32, 27, 26, 29, 28, 30, 31, 27, 32,
                28, 30, 27, 29, 31, 26, 32, 28, 29, 27,
                30, 28, 26, 32, 31, 30, 27, 29, 28, 30,
                31, 32, 27, 26, 29, 28, 30, 31, 27, 32
            ],
            'next_cycle_start': [
                30, 31, 28, 30, 32, 27, 33, 29, 30, 28,
                32, 30, 27, 33, 32, 31, 28, 30, 29, 31,
                33, 34, 28, 27, 31, 29, 31, 33, 28, 33,
                30, 31, 28, 30, 32, 27, 33, 29, 30, 28,
                32, 30, 27, 33, 32, 31, 28, 30, 29, 31,
                33, 34, 28, 27, 31, 29, 31, 33, 28, 33,
                30, 31, 28, 30, 32, 27, 33, 29, 30, 28,
                32, 30, 27, 33, 32, 31, 28, 30, 29, 31,
                33, 34, 28, 27, 31, 29, 31, 33, 28, 33
            ]
        }
        df = pd.DataFrame(data)
        X = df[['cycle_length']]
        y = df['next_cycle_start'] 
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

        model = LinearRegression()
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        print(f'Mean Squared Error: {mse}')
        print(f'R-squared: {r2}')

        user_cycle_length = request.cycle_length
        last_period_date_str = request.last_period_date
        last_period_date = datetime.strptime(last_period_date_str, "%Y-%m-%d")

        predicted_next_cycle_length = model.predict(np.array([[user_cycle_length]]))[0]
        predicted_next_cycle_start_date = last_period_date + timedelta(days=predicted_next_cycle_length)

        predicted_next_cycle_start_date_str = predicted_next_cycle_start_date.strftime("%Y-%m-%d")
        return {"predicted_date": predicted_next_cycle_start_date_str}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        print(f"Unexpected error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")