import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="whitegrid")


def plot_delays(df):
    plt.figure(figsize=(12, 5))
    sns.histplot(df["DEP_DELAY"], bins=80, kde=True, color="steelblue")
    plt.title("Rozkład opóźnień odlotów")
    plt.xlabel("Opóźnienie (min)")
    plt.ylabel("Liczba lotów")
    plt.savefig("../resources/plots/dep_delay.png")
    plt.show()

    plt.figure(figsize=(12, 5))
    sns.histplot(df["ARR_DELAY"], bins=80, kde=True, color="darkred")
    plt.title("Rozkład opóźnień przylotów")
    plt.xlabel("Opóźnienie (min)")
    plt.ylabel("Liczba lotów")
    plt.savefig("../resources/plots/arr_delay.png")
    plt.show()


def plot_monthly_delays(df):
    monthly = df.groupby("MONTH")["ARR_DELAY"].mean()

    plt.figure(figsize=(12, 5))
    sns.lineplot(x=monthly.index, y=monthly.values, marker="o")
    plt.title("Średnie opóźnienie przylotów w zależności od miesiąca")
    plt.xlabel("Miesiąc")
    plt.ylabel("Średnie opóźnienie (min)")
    plt.grid(True)
    plt.savefig("../resources/plots/monthly_delay.png")
    plt.show()


def plot_weather_relation(df):
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=df["O_TEMP"], y=df["DEP_DELAY"], alpha=0.3)
    plt.title("Temperatura na lotnisku wylotu vs opóźnienie odlotu")
    plt.xlabel("Temperatura (°C)")
    plt.ylabel("Opóźnienie (min)")
    plt.savefig("../resources/plots/temp_relation.png")
    plt.show()

    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=df["O_WSPD"], y=df["DEP_DELAY"], alpha=0.3, color="orange")
    plt.title("Prędkość wiatru na lotnisku wylotu vs opóźnienie odlotu")
    plt.xlabel("Wiatr (mph)")
    plt.ylabel("Opóźnienie (min)")
    plt.savefig("../resources/plots/wind_relation.png")
    plt.show()

    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=df["O_PRCP"], y=df["DEP_DELAY"], alpha=0.3, color="green")
    plt.title("Opady na lotnisku wylotu vs opóźnienie odlotu")
    plt.xlabel("Opady (in)")
    plt.ylabel("Opóźnienie (min)")
    plt.savefig("../resources/plots/precipitation_relation.png")
    plt.show()


def plot_carrier_delays(df):
    carrier = df.groupby("OP_CARRIER")["ARR_DELAY"].mean().sort_values()

    plt.figure(figsize=(14, 6))
    sns.barplot(
        x=carrier.index,
        y=carrier.values,
        hue=carrier.index,
        palette="viridis",
        legend=False
    )

    plt.title("Średnie opóźnienia przylotów wg przewoźnika")
    plt.xlabel("Przewoźnik")
    plt.ylabel("Średnie opóźnienie (min)")
    plt.xticks(rotation=45)
    plt.savefig("../resources/plots/carrier_delay.png")
    plt.show()


def plot_airports(df):
    plt.figure(figsize=(8, 6))
    plt.scatter(df["O_LONGITUDE"], df["O_LATITUDE"], s=1, alpha=0.3, label="Origin")
    plt.scatter(df["D_LONGITUDE"], df["D_LATITUDE"], s=1, alpha=0.3, label="Destination")
    plt.title("Położenie lotnisk w danych")
    plt.xlabel("Długość geograficzna")
    plt.ylabel("Szerokość geograficzna")
    plt.legend()
    plt.savefig("../resources/plots/airports.png")
    plt.show()


def plot_correlation(df):
    plt.figure(figsize=(14, 10))
    corr = df.corr(numeric_only=True)
    sns.heatmap(corr, cmap="coolwarm", annot=False)
    plt.title("Mapa korelacji zmiennych numerycznych")
    plt.savefig("../resources/plots/correlation.png")
    plt.show()


def plot_weekday_delays(df):
    weekday = df.groupby("DAY_OF_WEEK")["ARR_DELAY"].mean()

    plt.figure(figsize=(12, 5))
    sns.barplot(
        x=weekday.index,
        y=weekday.values,
        hue=weekday.index,
        palette="magma",
        legend=False
    )

    plt.title("Średnie opóźnienia przylotów wg dnia tygodnia")
    plt.xlabel("Dzień tygodnia (1=pon, 7=niedz)")
    plt.ylabel("Średnie opóźnienie (min)")
    plt.savefig("../resources/plots/weekday_delay.png")
    plt.show()


def create_all_plots(df):
    plot_delays(df)
    plot_monthly_delays(df)
    plot_weather_relation(df)
    plot_carrier_delays(df)
    plot_airports(df)
    plot_correlation(df)
    plot_weekday_delays(df)
