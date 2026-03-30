import pandas as pd
import json
import io

from visualize import create_all_plots


def df_to_json_info(df: pd.DataFrame) -> dict:
    info = {}

    info["shape"] = {
        "rows": df.shape[0],
        "columns": df.shape[1]
    }

    info["columns"] = list(df.columns)
    info["dtypes"] = df.dtypes.astype(str).to_dict()
    info["describe_numeric"] = df.describe(include="number").to_dict()
    info["describe_all"] = df.describe(include="all").to_dict()
    info["missing_values"] = df.isnull().sum().to_dict()
    info["unique_values"] = {
        col: df[col].nunique() for col in df.columns
    }
    info["sample_head"] = df.head().to_dict(orient="list")
    info["sample_tail"] = df.tail().to_dict(orient="list")

    buffer = io.StringIO()
    df.info(buf=buffer)
    info["info_text"] = buffer.getvalue()

    return info


def main():
    df = pd.read_csv("../resources/data/flight_with_weather_2016.csv")
    df.dropna(inplace=True)
    df_info = df_to_json_info(df)

    with open("../resources/info/df_info.json", "w", encoding="utf-8") as f:
        json.dump(df_info, f, indent=4, ensure_ascii=False)

    create_all_plots(df)


if __name__ == "__main__":
    main()
