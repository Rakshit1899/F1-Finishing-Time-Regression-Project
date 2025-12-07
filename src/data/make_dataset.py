import pandas as pd
from .load_data import load_raw_csv, save_processed

#############################################################
#  FINAL MODELING DATASET GENERATOR
#  Level: raceId × driverId (one row per driver per race)
#############################################################


def build_modeling_dataset() -> pd.DataFrame:
    # --------------------- Load raw data ---------------------
    races = load_raw_csv("races")
    results = load_raw_csv("results")
    drivers = load_raw_csv("drivers")
    constructors = load_raw_csv("constructors")
    circuits = load_raw_csv("circuits")
    pitstops = load_raw_csv("pit_stops")
    driver_standings = load_raw_csv("driver_standings")
    constructor_standings = load_raw_csv("constructor_standings")
    qualifying = load_raw_csv("qualifying")

    # --------------------- Base anchor (results) ---------------------
    df = results.copy()

    df = df.merge(
        races[["raceId", "year", "circuitId"]],
        on="raceId",
        how="left",
    )

    df = df.merge(
        drivers[["driverId", "driverRef", "code"]],
        on="driverId",
        how="left",
    )

    df = df.merge(
        constructors[["constructorId", "constructorRef"]],
        on="constructorId",
        how="left",
    )

    # --------------------- Constructor lineage mapping ---------------------
    constructor_map = {
        "Lotus F1": "Alpine",
        "Renault": "Alpine",
        "Alpine F1 Team": "Alpine",

        "Force India": "Aston Martin",
        "Racing Point": "Aston Martin",
        "Aston Martin": "Aston Martin",

        "Toro Rosso": "RB F1 Team",
        "AlphaTauri": "RB F1 Team",
        "RB F1 Team": "RB F1 Team",

        "Sauber": "Alfa Romeo",
        "Alfa Romeo": "Alfa Romeo",

        "Marussia": "Marussia",
        "Manor Marussia": "Marussia",
        "Caterham": "Caterham",

        "Ferrari": "Ferrari",
        "Red Bull": "Red Bull",
        "Mercedes": "Mercedes",
        "McLaren": "McLaren",
        "Williams": "Williams",
        "Haas F1 Team": "Haas",
        "Haas": "Haas",
    }
    df["constructorRef"] = df["constructorRef"].replace(constructor_map)

    # Circuits info
    df = df.merge(
        circuits[["circuitId", "circuitRef", "country", "alt"]],
        on="circuitId",
        how="left",
    )

    # Rename key columns from results
    df = df.rename(
        columns={
            "position": "final_position",
            "fastestLapTime": "fastest_lap_time",
            "milliseconds": "finishing_time_ms",
        }
    )

    # --------------------- Pitstop aggregation ---------------------
    df_pit = (
        pitstops.groupby(["raceId", "driverId"])
        .agg(
            stops_count=("stop", "count"),
            total_pit_ms=("milliseconds", "sum"),
            avg_pit_ms=("milliseconds", "mean"),
        )
        .reset_index()
    )

    df = df.merge(df_pit, on=["raceId", "driverId"], how="left")
    df[["stops_count", "total_pit_ms", "avg_pit_ms"]] = df[
        ["stops_count", "total_pit_ms", "avg_pit_ms"]
    ].fillna(0)

    # --------------------- Driver & constructor standings ---------------------
    df = df.merge(
        driver_standings[["raceId", "driverId", "points", "position"]].rename(
            columns={
                "points": "driver_standing_points",
                "position": "driver_standing_position",
            }
        ),
        on=["raceId", "driverId"],
        how="left",
    )

    df = df.merge(
        constructor_standings[
            ["raceId", "constructorId", "points", "position"]
        ].rename(
            columns={
                "points": "constructor_points",
                "position": "constructor_position",
            }
        ),
        on=["raceId", "constructorId"],
        how="left",
    )

    # --------------------- Qualifying data ---------------------
    # best available of q3 -> q2 -> q1
    qualifying["quali_time"] = (
        qualifying[["q3", "q2", "q1"]].bfill(axis=1).iloc[:, 0]
    )

    df = df.merge(
        qualifying[["raceId", "driverId", "position", "quali_time"]].rename(
            columns={"position": "quali_position"}
        ),
        on=["raceId", "driverId"],
        how="left",
    )

    # --------------------- Track attributes (lap length, distance, street) ---------------------
    lap_length_map = {
        "albert_park": 5.303,
        "sepang": 5.543,
        "bahrain": 5.412,
        "catalunya": 4.657,
        "istanbul": 5.338,
        "monaco": 3.337,
        "villeneuve": 4.361,
        "magny_cours": 4.411,
        "silverstone": 5.891,
        "hockenheimring": 4.575,
        "hungaroring": 4.381,
        "valencia": 5.419,
        "spa": 7.004,
        "monza": 5.793,
        "marina_bay": 5.063,
        "fuji": 4.563,
        "shanghai": 5.451,
        "interlagos": 4.309,
        "indianapolis": 4.192,
        "nurburgring": 5.148,
        "imola": 4.909,
        "suzuka": 5.807,
        "vegas": 6.201,
        "yas_marina": 5.554,
        "galvez": 4.259,
        "jerez": 4.428,
        "estoril": 4.182,
        "okayama": 3.703,
        "adelaide": 3.780,
        "kyalami": 4.529,
        "donington": 4.023,
        "rodriguez": 4.304,
        "phoenix": 3.800,
        "ricard": 5.842,
        "yeongam": 5.615,
        "jacarepagua": 5.031,
        "detroit": 4.000,
        "brands_hatch": 4.207,
        "zandvoort": 4.259,
        "dijon": 3.801,
        "dallas": 3.900,
        "long_beach": 3.167,
        "las_vegas": 3.650,
        "jarama": 3.404,
        "watkins_glen": 5.435,
        "anderstorp": 4.018,
        "mosport": 3.957,
        "montjuic": 3.790,
        "nivelles": 3.720,
        "charade": 8.055,
        "tremblant": 4.265,
        "essarts": 6.542,
        "lemans": 13.626,
        "reims": 8.302,
        "george": 3.920,
        "zeltweg": 3.200,
        "aintree": 4.828,
        "boavista": 7.400,
        "riverside": 5.210,
        "avus": 8.300,
        "monsanto": 5.440,
        "sebring": 5.200,
        "ain-diab": 7.600,
        "pescara": 25.800,
        "bremgarten": 7.280,
        "pedralbes": 6.316,
        "buddh": 5.125,
        "americas": 5.513,
        "red_bull_ring": 4.318,
        "sochi": 5.848,
        "baku": 6.003,
        "portimao": 4.653,
        "mugello": 5.245,
        "jeddah": 6.174,
        "losail": 5.380,
        "miami": 5.410,
    }

    df["lap_length_km"] = df["circuitRef"].map(lap_length_map)
    df["total_distance_km"] = df["lap_length_km"] * df["laps"]

    street_ids = [
        "monaco",
        "marina_bay",
        "miami",
        "jeddah",
        "baku",
        "vegas",
        "las_vegas",
        "phoenix",
        "adelaide",
        "detroit",
        "dallas",
        "long_beach",
        "valencia",
    ]
    df["is_street"] = df["circuitRef"].isin(street_ids).astype(int)

    # --------------------- Rolling form (last 10 races, points-based) ---------------------
    df["driver_standing_points"] = pd.to_numeric(
        df["driver_standing_points"], errors="coerce"
    ).fillna(0)
    df["constructor_points"] = pd.to_numeric(
        df["constructor_points"], errors="coerce"
    ).fillna(0)

    df = df.sort_values(["year", "raceId", "driverId", "constructorRef"])

    df["driver_points_last10"] = (
        df.groupby("driverId")["driver_standing_points"]
        .rolling(10, min_periods=1)
        .sum()
        .reset_index(level=0, drop=True)
    )

    df["constructor_points_last10"] = (
        df.groupby("constructorRef")["constructor_points"]
        .rolling(10, min_periods=1)
        .sum()
        .reset_index(level=0, drop=True)
    )

    # --------------------- Filter for hybrid era ---------------------
    df = df[df["year"] >= 2013]

    # --------------------- Convert time-like fields to ms ---------------------
    df["fastest_lap_time"] = df["fastest_lap_time"].apply(time_to_ms)
    df["quali_time"] = df["quali_time"].apply(time_to_ms)

    # Drop rows missing finishing time (DNF / NC timings)
    # ----------------------------------------
    df = df[df["finishing_time_ms"].notna()]
    # --------------------- Final export struct ---------------------
    final_cols = [
        "year",
        "raceId",
        "circuitId",
        "circuitRef",
        "country",
        "statusId",
        "driverId",
        "driverRef",
        "constructorId",
        "code",
        "constructorRef",      

 
        
        "grid",
        "final_position",
        "laps",
        "fastest_lap_time",
        "finishing_time_ms",
        "stops_count",
        "total_pit_ms",
        "avg_pit_ms",
        "driver_standing_points",
        "driver_standing_position",
        "constructor_points",
        "constructor_position",
        "quali_position",
        "quali_time",
        "alt",
        "lap_length_km",
        "total_distance_km",
        "is_street",
        "driver_points_last10",
        "constructor_points_last10",
    ]

    df_model = df[final_cols].copy()
    df_model = enforce_dtypes(df_model)

    # save (no extension, as per your convention)
    save_processed(df_model, "f1_modeling_dataset_final")
    return df_model


#############################################################
#  Helpers
#############################################################


def time_to_ms(x):
    """Convert 'M:SS.sss' strings to milliseconds; leave others as-is."""
    if isinstance(x, str) and ":" in x:
        try:
            m, s = x.split(":")
            return int(m) * 60000 + float(s) * 1000
        except ValueError:
            return pd.NA
    return x


def enforce_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Cast ideal datatypes for memory efficiency & ML compatibility."""

    # Identifiers / keys
    id_cols_32 = ["raceId", "driverId", "constructorId", "circuitId", "year"]
    for col in id_cols_32:
        df[col] = pd.to_numeric(df[col], errors="coerce").astype("int32")

    # Ordered integer features
    int16_cols = [
        "grid",
        "final_position",
        "laps",
        "driver_standing_position",
        "constructor_position",
        "quali_position",
    ]
    for col in int16_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype("int16")

    # Boolean / binary
    df["is_street"] = df["is_street"].astype("int8")

    # Continuous numeric
    float_cols = [
        "finishing_time_ms",
        "fastest_lap_time",
        "total_pit_ms",
        "avg_pit_ms",
        "driver_standing_points",
        "constructor_points",
        "quali_time",
        "alt",
        "lap_length_km",
        "total_distance_km",
        "driver_points_last10",
        "constructor_points_last10",
    ]
    for col in float_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce").astype("float64")

    # Categoricals
    cat_cols = ["driverRef", "code", "constructorRef", "circuitRef", "country"]
    for col in cat_cols:
        df[col] = df[col].astype("category")

    return df
