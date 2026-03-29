# FlightDelayPredictor


## Large File Tracking
Use `git-lfs` tool for files that are too large for git repo. Install it from: https://git-lfs.com/ Then follow the guide on the main page there.

## Notice about weather API
You'll probably get the `429` error after you've reached 10_000 requests or spamed too much in a row. For that script will wait 1 minute before proceeding. The data is cached under
`weather_cache_master.pkl` so progress is tracked and in the next run it will just get some data from cache. It's possible that when you're not using the cached file you'll need to run it few times before you get the "correct data". The more airports you have the longer it takes.
Weather API: https://open-meteo.com/en/docs