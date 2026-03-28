# FlightDelayPredictor




## Notice about weather API
You'll probably get the `429` error after you've reached 10_000 requests or spamed too much in a row. For that script will wait 1 minute before proceeding. The data is cached under
`weather_cache_master.pkl` so progress is tracked and in the next run it will just get some data from cache. It's possible that when you're not using the cached file you'll need to run it few times before you get the "correct data". The more airports you have the longer it takes.