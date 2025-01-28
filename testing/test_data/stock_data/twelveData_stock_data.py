import requests
import pandas as pd

# Define the API endpoint and parameters
url = "https://api.twelvedata.com/time_series"
params = {
    "apikey": "6cbd351386a54edd90f7a34ff4129cd9",
    "interval": "5min",
    "symbol": "AMD",
    "outputsize": 5000,
    "end_date": "2023-06-30 11:25:00",  # Corrected the end_date format
    "dp": "8"
}

# Make the API request
response = requests.get(url, params=params)

# Check if the request was successful
if response.status_code == 200:
    data = response.json()
    
    # Check if 'values' key is in the response data
    if 'values' in data:
        # Convert the list of data points into a DataFrame
        df = pd.DataFrame(data['values'])
        
        # Convert the 'datetime' column to datetime objects
        df['datetime'] = pd.to_datetime(df['datetime'])
        
        # Sort the DataFrame by datetime in descending order
        df.sort_values('datetime', ascending=False, inplace=True)
        
        # Reset the index of the DataFrame
        df.reset_index(drop=True, inplace=True)
        
        # Save the DataFrame to a CSV file with ';' as the separator
        filename = 'time-series-AMD-5min(2).csv'
        df.to_csv(filename, sep=';', index=False)
        print(f"Data saved to {filename}")
    else:
        print("Error: 'values' key not found in the response data")
else:
    print(f"Error: Failed to fetch data. Status code: {response.status_code}")
    print(response.text)
