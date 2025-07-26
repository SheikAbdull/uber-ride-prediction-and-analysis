# data_generator.py
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def generate_dummy_data(rows=1000):
    np.random.seed(42)
    data = {
        'pickup_datetime': [datetime.now() - timedelta(minutes=np.random.randint(100000)) for _ in range(rows)],
        'pickup_longitude': -73.95 + np.random.rand(rows) * 0.1,
        'pickup_latitude': 40.75 + np.random.rand(rows) * 0.1,
        'dropoff_longitude': -73.95 + np.random.rand(rows) * 0.1,
        'dropoff_latitude': 40.75 + np.random.rand(rows) * 0.1,
        'passenger_count': np.random.randint(1, 5, rows),
        'fare_amount': np.round(np.random.uniform(5, 50, rows), 2)
    }
    df = pd.DataFrame(data)
    df.to_csv("uber_data.csv", index=False)

generate_dummy_data()

