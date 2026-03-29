import pandas as pd
import os
from datetime import datetime
from weather_utils import get_weather_for_flights, export_master_weather,inject_weather

MASTER_PARQUET_PATH = "../../data/kaggle/master_weather_2019_2023.parquet"

def sync_weather_repository(flights_df):
    """
    Używa pliku Parquet jako głównej bazy danych. 
    Sprawdza braki, dociąga je i aktualizuje plik Parquet.
    """
    print("--- Synchronizacja z bazą Parquet ---")
    
    needed_origins = flights_df['ORIGIN'].unique()
    start_needed = pd.to_datetime(flights_df['FL_DATE']).min().tz_localize(None)
    end_needed = pd.to_datetime(flights_df['FL_DATE']).max().tz_localize(None)

    if os.path.exists(MASTER_PARQUET_PATH):
        master_df = pd.read_parquet(MASTER_PARQUET_PATH)
        master_df['time'] = pd.to_datetime(master_df['time']).dt.tz_localize(None)
        cached_origins = master_df['ORIGIN_KEY'].unique()
        m_min, m_max = master_df['time'].min(), master_df['time'].max()
        
        missing_airports = set(needed_origins) - set(cached_origins)
        needs_update = (
            len(missing_airports) > 0 or 
            start_needed < m_min or 
            end_needed > m_max
        )
        
        if not needs_update:
            print("Wszystko jest w Parquet. Łączę dane...")
            return inject_weather(flights_df, MASTER_PARQUET_PATH)
    else:
        print("Brak pliku Parquet. Tworzę nową bazę...")
        needs_update = True
    if needs_update:
        print("Uruchamiam downloader API dla brakujących danych...")
        df_enriched = get_weather_for_flights(flights_df, force_update=True)
        export_master_weather(df_enriched, MASTER_PARQUET_PATH)
        
        return df_enriched

def clean_delay_columns(df):
    """
    Czyści NaN w kolumnach opóźnień.
    """
    delay_cols = [
        'DELAY_DUE_CARRIER', 'DELAY_DUE_WEATHER', 'DELAY_DUE_NAS', 
        'DELAY_DUE_SECURITY', 'DELAY_DUE_LATE_AIRCRAFT'
    ]
    available = [c for c in delay_cols if c in df.columns]
    if available:
        df[available] = df[available].fillna(0.0)
        print(f"Wypełniono zerami kolumny opóźnień: {len(available)} kolumn.")
    return df