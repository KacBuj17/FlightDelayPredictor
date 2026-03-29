import pandas as pd
import os
from datetime import datetime
from weather_utils import get_weather_for_flights

MASTER_CACHE_PATH = "../../data/kaggle/weather_cache_master.pkl"

def sync_weather_repository(flights_df):
    """
    Sprawdza, czy dla lotnisk i dat w flights_df mamy już dane w cache.
    """
    print("--- Synchronizacja Repozytorium Pogodowego ---")
    
    needed_origins = flights_df['ORIGIN'].unique()
    start_date_needed = pd.to_datetime(flights_df['FL_DATE'], utc=True).min().tz_localize(None)
    end_date_needed = pd.to_datetime(flights_df['FL_DATE'], utc=True).max().tz_localize(None)
    
    if os.path.exists(MASTER_CACHE_PATH):
        cache_df = pd.read_pickle(MASTER_CACHE_PATH)
        
        if not cache_df.empty:
            cached_origins = cache_df['ORIGIN_KEY'].unique()
            cache_times = pd.to_datetime(cache_df['time'], utc=True).dt.tz_localize(None)
            cache_min = cache_times.min()
            cache_max = cache_times.max()
            
            missing_airports = set(needed_origins) - set(cached_origins)
            needs_update = (
                len(missing_airports) > 0 or 
                start_date_needed < cache_min or 
                end_date_needed > cache_max
            )
            
            if not needs_update:
                print("Pomyślnie zweryfikowano: Wszystkie dane są już w cache. Pomijam pobieranie.")
                return get_weather_for_flights(flights_df, force_update=False)
            else:
                if missing_airports:
                    print(f"Znaleziono nowe lotniska: {list(missing_airports)}")
                if start_date_needed < cache_min or end_date_needed > cache_max:
                    print(f"Zakres w cache: {cache_min.date()} do {cache_max.date()}")
                    print(f"Zakres potrzebny: {start_date_needed.date()} do {end_date_needed.date()}")
        else:
            print("Cache jest pusty.")
    
    print("Uruchamiam proces uzupełniania danych przez API...")
    updated_df = get_weather_for_flights(flights_df, force_update=True)
    
    return updated_df

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