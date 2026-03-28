import pandas as pd
import requests
import time
import os
import openmeteo_requests
import requests_cache
from retry_requests import retry

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
def get_weather_for_flights(flights_df, force_update=False):
    weather_columns = ['temperature_2m', 'precipitation', 'wind_speed_10m']
    
    # Jeśli wszystkie kluczowe kolumny już są i nie wymuszamy aktualizacji...
    if all(col in flights_df.columns for col in weather_columns) and not force_update:
        print("Dane pogodowe są już obecne w datasetu. Pomijam merge.")
        return flights_df
    
    # 1. Przygotowanie danych i klienta API
    df = flights_df.copy()
    airports_df = load_airport_data()
    
    # Konfiguracja oficjalnego SDK Open-Meteo z lokalnym cache zapytan
    cache_session = requests_cache.CachedSession('.cache', expire_after=-1)
    retry_session = retry(cache_session, retries=5, backoff_factor=2)
    openmeteo = openmeteo_requests.Client(session=retry_session)
    
    df['ORIGIN'] = df['ORIGIN'].astype(str).str.strip()
    df = df.merge(airports_df, left_on='ORIGIN', right_on='IATA_CODE', how='left')
    
    unique_locs = df[['ORIGIN', 'LATITUDE', 'LONGITUDE']].dropna().drop_duplicates().copy()
    unique_locs['ORIGIN'] = unique_locs['ORIGIN'].astype(str).str.strip()
    
    start_date = pd.to_datetime(df['FL_DATE']).min().strftime('%Y-%m-%d')
    end_date = pd.to_datetime(df['FL_DATE']).max().strftime('%Y-%m-%d')
    
    cache_dir = "../../data/kaggle"
    cache_file = os.path.join(cache_dir, "weather_cache_master.pkl")
    if not os.path.exists(cache_dir): os.makedirs(cache_dir, exist_ok=True)

    # 2. Wczytywanie Cache
    all_weather_dfs = []
    done_origins = set()

    if os.path.exists(cache_file):
        try:
            cache_df = pd.read_pickle(cache_file)
            if not cache_df.empty:
                cache_df['ORIGIN_KEY'] = cache_df['ORIGIN_KEY'].astype(str).str.strip()
                all_weather_dfs.append(cache_df)
                done_origins = set(cache_df['ORIGIN_KEY'].unique())
                print(f"--- CACHE: Wczytano {len(done_origins)} lotnisk ---")
        except Exception as e:
            print(f"Błąd odczytu cache: {e}")

    # 3. Filtrowanie
    to_process = unique_locs[~unique_locs['ORIGIN'].isin(done_origins)]
    total_to_download = len(to_process)
    
    if total_to_download == 0:
        print("Wszystko jest w cache. Łączę dane...")
    else:
        print(f"Do pobrania: {total_to_download} / {len(unique_locs)} lotnisk.")

    # 4. Pętla pobierania (SDK Style)
    url = "https://archive-api.open-meteo.com/v1/archive"
    
    for i, (_, row) in enumerate(to_process.iterrows()):
        origin = row['ORIGIN']
        
        params = {
            "latitude": row['LATITUDE'],
            "longitude": row['LONGITUDE'],
            "start_date": start_date,
            "end_date": end_date,
            "hourly": ["temperature_2m", "relative_humidity_2m", "precipitation", 
                       "cloud_cover", "wind_speed_10m", "surface_pressure"],
            "timezone": "UTC"
        }

        try:
            # Pobieranie przez SDK (binarnie)
            responses = openmeteo.weather_api(url, params=params)
            response = responses[0]

            # Procesowanie danych godzinowych
            hourly = response.Hourly()
            hourly_data = {"time": pd.date_range(
                start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
                end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
                freq=pd.Timedelta(seconds=hourly.Interval()),
                inclusive="left"
            )}
            
            for idx, var in enumerate(params["hourly"]):
                hourly_data[var] = hourly.Variables(idx).ValuesAsNumpy()

            w_df = pd.DataFrame(data=hourly_data)
            w_df['ORIGIN_KEY'] = origin
            all_weather_dfs.append(w_df)

            # --- ATOMIC SAFE WRITE ---
            # Zawsze łączymy wszystko co mamy w pamięci, żeby nie zgubić danych
            current_full_weather = pd.concat(all_weather_dfs, ignore_index=True).drop_duplicates()
            current_full_weather.to_pickle(cache_file + ".tmp")
            os.replace(cache_file + ".tmp", cache_file)
            
            print(f"[{i+1}/{total_to_download}] OK: {origin}")
            time.sleep(1.5) # SDK jest szybkie, ale szanujemy limity API

        except Exception as e:
            if "429" in str(e):
                print("Błąd 429: Limit przekroczony. Przerywam, by nie pogarszać sytuacji.")
                break
            print(f"Błąd przy {origin}: {e}")
            continue

    # 5. Przygotowanie danych pogodowych
    if not all_weather_dfs: 
        return df
    
    weather_full = pd.concat(all_weather_dfs, ignore_index=True).drop_duplicates()
    
    weather_full['time'] = pd.to_datetime(weather_full['time'], utc=True).dt.tz_localize(None)
    weather_full['ORIGIN_KEY'] = weather_full['ORIGIN_KEY'].astype(str).str.strip()

    # 7. MERGE Z GŁÓWNYM DATAFRAME
    print("Przygotowanie kluczy czasowych w lotach...")
    
    df['time_str'] = df['CRS_DEP_TIME'].astype(int).astype(str).str.zfill(4).replace('2400', '0000')
    
    df['scheduled_datetime'] = pd.to_datetime(
        pd.to_datetime(df['FL_DATE']).dt.strftime('%Y-%m-%d') + ' ' + 
        df['time_str'].str[:2] + ':' + df['time_str'].str[2:], 
        errors='coerce',
        utc=True #
    ).dt.tz_localize(None) 
    
    df['weather_key'] = df['scheduled_datetime'].dt.round('h')

    print(f"Typ klucza loty: {df['weather_key'].dtype}")
    print(f"Typ klucza pogoda: {weather_full['time'].dtype}")

    final_df = df.merge(
        weather_full, 
        left_on=['ORIGIN', 'weather_key'], 
        right_on=['ORIGIN_KEY', 'time'], 
        how='left'
    )

    print("Łączę dane (Merge)...")
    final_df = df.merge(
        weather_full, 
        left_on=['ORIGIN', 'weather_key'], 
        right_on=['ORIGIN_KEY', 'time'], 
        how='left'
    )

    # Sprzątanie kolumn pomocniczych
    drop_cols = ['IATA_CODE', 'ORIGIN_KEY', 'time', 'weather_key', 'time_str', 
                 'LATITUDE', 'LONGITUDE', 'scheduled_datetime']
    final_df.drop(columns=[c for c in drop_cols if c in final_df.columns], inplace=True)

    print(f"Merge zakończony. Sukces dopasowania: {final_df['temperature_2m'].notna().mean():.2%}")
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
    
'''
Save the final weather dataset. Weather API has limits for the requests so we want to have it in some file.
'''
def save_and_optimize_weather_ds(df, filename="../../data/kaggle/flights_weather_final.parquet"):
    print(f"Rozpoczynam optymalizację datasetu o rozmiarze: {len(df)} wierszy.")
    df_opt = df.copy()
    float_cols = df_opt.select_dtypes(include=['float64']).columns
    df_opt[float_cols] = df_opt[float_cols].astype('float32')
    
    # Opóźnienia czy numery lotów nie potrzebują 64-bitów.
    int_cols = df_opt.select_dtypes(include=['int64']).columns
    for col in int_cols:
        col_max = df_opt[col].max()
        col_min = df_opt[col].min()
        if col_max < 32767 and col_min > -32768:
            df_opt[col] = df_opt[col].astype('int16')
        else:
            df_opt[col] = df_opt[col].astype('int32')

    # ORIGIN, DEST, CARRIER powtarzają się tysiące razy - 'category' drastycznie zmniejsza wagę pliku.
    obj_cols = df_opt.select_dtypes(include=['object']).columns
    for col in obj_cols:
        # Jeśli unikalnych wartości jest mało (< 50% wszystkich wierszy), konwertuj na kategorię
        if df_opt[col].nunique() < len(df_opt) * 0.5:
            df_opt[col] = df_opt[col].astype('category')

    # Upewnienie się, że daty są poprawne
    date_cols = [c for c in df_opt.columns if 'DATE' in c.upper() or 'time' in c.lower()]
    for col in date_cols:
        df_opt[col] = pd.to_datetime(df_opt[col])


    print(f"Zapisywanie do formatu Parquet: {filename}")
    try:
        # Używamy snappy dla balansu między szybkością a rozmiarem
        df_opt.to_parquet(filename, engine='pyarrow', compression='snappy', index=False)
        
        file_size = os.path.getsize(filename) / (1024 * 1024)
        print(f"--- SUKCES ---")
        print(f"Finalny rozmiar pliku: {file_size:.2f} MB")
        
        # Porównanie zużycia pamięci RAM
        old_mem = df.memory_usage(deep=True).sum() / (1024**2)
        new_mem = df_opt.memory_usage(deep=True).sum() / (1024**2)
        print(f"Zużycie RAM: {old_mem:.1f} MB -> {new_mem:.1f} MB (Redukcja: {100*(1-new_mem/old_mem):.1f}%)")
        
    except Exception as e:
        print(f"Błąd podczas zapisu: {e}")
        
    return df_opt


# Export
def export_master_weather(df, output_path="../../data/kaggle/master_weather_2019_2023.parquet"):
    """
    Wyodrębnia unikalną bazę pogodową z połączonego datasetu lotów.
    """
    print("Przygotowanie bazy referencyjnej pogody...")
    
    # 1. Definiujemy kolumny, które chcemy zachować w bazie pogodowej
    # Mapujemy Twoje kolumny na standardowe nazwy dla bazy master
    weather_mapping = {
        'ORIGIN': 'ORIGIN_KEY',
        'weather_hour_key': 'time'
    }
    
    # Kolumny z danymi pogodowymi (te, które już masz w DF)
    data_cols = [
        'temperature_2m', 'relative_humidity_2m', 'precipitation', 
        'cloud_cover', 'wind_speed_10m', 'surface_pressure'
    ]
    
    # 2. Wybieramy potrzebne dane i zmieniamy nazwy na uniwersalne
    # Wybieramy ORIGIN, weather_hour_key oraz wszystkie dane pogodowe
    cols_to_extract = ['ORIGIN', 'weather_hour_key'] + data_cols
    master_weather = df[cols_to_extract].copy()
    master_weather = master_weather.rename(columns=weather_mapping)
    
    # 3. USUWAMY DUPLIKATY
    # Ponieważ w locie mogło być 1000 samolotów z JFK o 14:00, 
    # musimy zostawić tylko jeden wpis pogodowy dla tej pary (miejsce, czas).
    initial_rows = len(master_weather)
    master_weather = master_weather.drop_duplicates(subset=['ORIGIN_KEY', 'time'])
    
    print(f"Zredukowano {initial_rows} wierszy lotów do {len(master_weather)} unikalnych wpisów pogodowych.")
    
    # 4. Optymalizacja typów przed zapisem
    # Float64 -> Float32 (oszczędność 50% miejsca)
    for col in data_cols:
        if col in master_weather.columns:
            master_weather[col] = master_weather[col].astype('float32')
            
    # ORIGIN_KEY jako kategoria (bardzo ważne dla RAM)
    master_weather['ORIGIN_KEY'] = master_weather['ORIGIN_KEY'].astype('category')
    
    # 5. Zapis do Parquet
    master_weather.to_parquet(output_path, engine='pyarrow', compression='snappy', index=False)
    
    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"--- SUKCES ---")
    print(f"Plik master_weather zapisany: {output_path}")
    print(f"Rozmiar: {size_mb:.2f} MB")
    
    return master_weather

def inject_weather(flights_df, master_weather_path="../../data/kaggle/master_weather_2019_2023.parquet"):
    # 1. Wczytaj loty i master weather
    weather_ref = pd.read_parquet(master_weather_path)
    
    # 2. Przygotuj klucz czasu w lotach (zaokrąglony do godziny, UTC)
    flights_df['weather_key'] = pd.to_datetime(
        flights_df['FL_DATE'].astype(str) + ' ' + 
        flights_df['CRS_DEP_TIME'].astype(int).astype(str).str.zfill(4).str[:2] + ':00:00'
    )
    
    # 3. Szybki Merge
    enriched_df = flights_df.merge(
        weather_ref,
        left_on=['ORIGIN', 'weather_key'],
        right_on=['ORIGIN_KEY', 'time'],
        how='left'
    )
    
    return enriched_df.drop(columns=['ORIGIN_KEY', 'time', 'weather_key'])