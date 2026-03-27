import pandas as pd
from datetime import datetime, timedelta
import requests
import time
import os

'''
Will load airport data needed for our `flights` dataset in which we only have the Aiport code without geo localization.
'''
def load_airport_data():
    url = "https://raw.githubusercontent.com/jpatokal/openflights/master/data/airports.dat"
    cols = ["ID", "Name", "City", "Country", "IATA", "ICAO", "LATITUDE", "LONGITUDE", 
            "Altitude", "Timezone", "DST", "Tz_db", "Type", "Source"]
    airports_df = pd.read_csv(url, names=cols, na_values="\\N")
    
    # Czyścimy: tylko rekordy z kodem IATA i współrzędnymi
    airports_df = airports_df[["IATA", "LATITUDE", "LONGITUDE"]].dropna(subset=["IATA"])
    airports_df = airports_df.rename(columns={"IATA": "IATA_CODE"})
    
    # Usuwamy ewentualne spacje
    airports_df["IATA_CODE"] = airports_df["IATA_CODE"].str.strip()
    return airports_df

'''
Gets weather for the flights
'''
def get_weather_for_flights(flights_df):
    df = flights_df.copy()
    airports_df = load_airport_data()
    
    df['ORIGIN'] = df['ORIGIN'].astype(str).str.strip()
    df = df.merge(airports_df, left_on='ORIGIN', right_on='IATA_CODE', how='left')
    
    unique_locs = df[['ORIGIN', 'LATITUDE', 'LONGITUDE']].dropna().drop_duplicates()
    
    start_date = pd.to_datetime(df['FL_DATE']).min().strftime('%Y-%m-%d')
    end_date = pd.to_datetime(df['FL_DATE']).max().strftime('%Y-%m-%d')
    
    print(f"Pobieranie: {len(unique_locs)} lotnisk, {start_date} do {end_date}")
    
    cache_dir = "../../data/kaggle"
    cache_file = os.path.join(cache_dir, "weather_cache_master.pkl")
    
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir, exist_ok=True)

    all_weather_dfs = []
    done_origins = []

    if os.path.exists(cache_file):
        try:
            cache_df = pd.read_pickle(cache_file)
            all_weather_dfs.append(cache_df)
            done_origins = cache_df['ORIGIN_KEY'].unique().tolist()
            print(f"Cache: {len(done_origins)} lotnisk")
        except Exception:
            pass

    for _, row in unique_locs.iterrows():
        origin = row['ORIGIN']
        if origin in done_origins:
            continue
        
        url = "https://archive-api.open-meteo.com/v1/archive"
        params = {
            "latitude": row['LATITUDE'],
            "longitude": row['LONGITUDE'],
            "start_date": start_date,
            "end_date": end_date,
            "hourly": "temperature_2m,relative_humidity_2m,precipitation,cloud_cover,wind_speed_10m,surface_pressure",
            "timezone": "UTC"
        }
        
        try:
            response = requests.get(url, params=params, timeout=15)
            
            if response.status_code == 429:
                print("Status 429: Czekam 60s")
                time.sleep(60)
                response = requests.get(url, params=params)
            
            if response.status_code == 200:
                data = response.json()
                w_df = pd.DataFrame(data['hourly'])
                w_df['time'] = pd.to_datetime(w_df['time'])
                w_df['ORIGIN_KEY'] = origin
                all_weather_dfs.append(w_df)
                
                pd.concat(all_weather_dfs, ignore_index=True).to_pickle(cache_file)
                print(f"Zapisano: {origin}")
                time.sleep(2.0)
            else:
                print(f"Blad {origin}: {response.status_code}")
                time.sleep(2.0)
                
        except Exception as e:
            print(f"Blad sieci {origin}: {e}")
            break

    if not all_weather_dfs:
        return df

    weather_full = pd.concat(all_weather_dfs, ignore_index=True).drop_duplicates()

    df['time_str'] = df['CRS_DEP_TIME'].astype(int).astype(str).str.zfill(4).replace('2400', '0000')
    df['scheduled_datetime'] = pd.to_datetime(
        pd.to_datetime(df['FL_DATE']).dt.strftime('%Y-%m-%d') + ' ' + 
        df['time_str'].str[:2] + ':' + df['time_str'].str[2:], 
        errors='coerce'
    )
    df['weather_key'] = df['scheduled_datetime'].dt.round('h')

    final_df = df.merge(
        weather_full, 
        left_on=['ORIGIN', 'weather_key'], 
        right_on=['ORIGIN_KEY', 'time'], 
        how='left'
    )

    drop_cols = ['IATA_CODE', 'ORIGIN_KEY', 'time', 'weather_key', 'time_str', 'LATITUDE', 'LONGITUDE']
    final_df = final_df.drop(columns=[c for c in drop_cols if c in final_df.columns])

    return final_df

'''
It will retry to get the weather data for the airport by `airport_code`
'''
def retry_weather_for_airport(airport_code, flights_df):
    airports_df = load_airport_data()
    row = airports_df[airports_df['IATA_CODE'] == airport_code].iloc[0]
    
    start_date = pd.to_datetime(flights_df['FL_DATE']).min().strftime('%Y-%m-%d')
    end_date = pd.to_datetime(flights_df['FL_DATE']).max().strftime('%Y-%m-%d')
    
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": row['LATITUDE'],
        "longitude": row['LONGITUDE'],
        "start_date": start_date,
        "end_date": end_date,
        "hourly": "temperature_2m,relative_humidity_2m,precipitation,cloud_cover,wind_speed_10m,surface_pressure",
        "timezone": "UTC"
    }
    
    print(f"Ponawiam probe dla {airport_code}...")
    try:
        response = requests.get(url, params=params, timeout=20)
        if response.status_code == 200:
            data = response.json()
            w_df = pd.DataFrame(data['hourly'])
            w_df['time'] = pd.to_datetime(w_df['time'])
            w_df['ORIGIN_KEY'] = airport_code
            
            cache_file = "../../data/kaggle/weather_cache_master.pkl"
            if os.path.exists(cache_file):
                old_cache = pd.read_pickle(cache_file)
                old_cache = old_cache[old_cache['ORIGIN_KEY'] != airport_code]
                new_cache = pd.concat([old_cache, w_df], ignore_index=True)
                new_cache.to_pickle(cache_file)
            else:
                w_df.to_pickle(cache_file)
                
            print(f"Sukces! Dane dla {airport_code} odswiezone w cache.")
            return w_df
        else:
            print(f"Blad {response.status_code}")
    except Exception as e:
        print(f"Blad: {e}")


'''
Sometimes you'll need to delete `invalid` data from cache that's the method you want to use.
'''
def remove_airport_from_cache(airport_code, cache_path="../../data/kaggle/weather_cache_master.pkl"):
    if not os.path.exists(cache_path):
        print(f"Plik cache nie istnieje: {cache_path}")
        return
    
    try:
        cache_df = pd.read_pickle(cache_path)
        original_count = len(cache_df)
        
        cache_df = cache_df[cache_df['ORIGIN_KEY'] != airport_code]
        
        if len(cache_df) < original_count:
            cache_df.to_pickle(cache_path)
            print(f"Sukces: Usunieto {airport_code} z cache. (Usunieto {original_count - len(cache_df)} rekordow)")
        else:
            print(f"Lotniska {airport_code} nie bylo w pliku cache.")
            
    except Exception as e:
        print(f"Blad podczas modyfikacji cache: {e}")
        
        
def list_cached_airports(cache_path="../../data/kaggle/weather_cache_master.pkl"):
    if not os.path.exists(cache_path):
        print("Cache jeszcze nie istnieje.")
        return []
    
    try:
        cache_df = pd.read_pickle(cache_path)
        if 'ORIGIN_KEY' in cache_df.columns:
            unique_airports = cache_df['ORIGIN_KEY'].unique().tolist()
            print(f"--- Stan Cache ---")
            print(f"Liczba lotnisk: {len(unique_airports)}")
            print(f"Lotniska: {', '.join(sorted(unique_airports))}")
            print(f"Laczna liczba wierszy danych: {len(cache_df)}")
            return unique_airports
        else:
            print("Plik cache istnieje, ale nie ma kolumny 'ORIGIN_KEY'.")
            return []
    except Exception as e:
        print(f"Blad odczytu cache: {e}")
        return []