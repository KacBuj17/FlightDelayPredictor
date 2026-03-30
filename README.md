# FlightDelayPredictor
Setup instructions:

Install python and check the version. Use the latest one. Setup .venv or other option you prefer.
```
python --version
python -m venv .venv
source .venv/bin/activate #or similar command for other shells
```

Installing dependencies:
```
pip install -r requirements.txt
```

## Large File Tracking
Use `git-lfs` tool for files that are too large for git repo. Install it from: https://git-lfs.com/ Then follow the guide on the main page there.

## Notice about weather API
You'll probably get the `429` error after you've reached 10_000 requests or spamed too much in a row. For that script will wait 1 minute before proceeding. The data is cached under
`weather_cache_master.pkl` so progress is tracked and in the next run it will just get some data from cache. It's possible that when you're not using the cached file you'll need to run it few times before you get the "correct data". The more airports you have the longer it takes.
Weather API: https://open-meteo.com/en/docs

## Notice about Airport API
My dataset doesn't contain the info about the airports coords. So I fetch the data from: https://raw.githubusercontent.com/jpatokal/openflights/master/data/airports.dat

## Info about weather related parquet files
As the Weather API has request limits and also it takes hours to fetch the data per timestamp. The `master-weather_2019_2023.parquet` holds the values for Airports. If you need to fetch data for new timestamp use: `sync_weather_repository()` function from weather_manager file. It will merge the new data into the existing parquet file. But first it checks if it's not already present there.

https://github.com/Flnny/Delay-data

To run the notebook successfully, please download the dataset from the following source:
flight_event_data

After downloading, extract the contents and place all files into the data/ directory in the project root.

The directory structure should look like this:

data/
├── airlines.csv
├── airports.csv
├── cancellation_codes.csv
└── flights.csv

