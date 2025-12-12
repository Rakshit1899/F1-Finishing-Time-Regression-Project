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

    df = df.merge(races[["raceId","year","circuitId"]], on="raceId", how="left")
    df = df.merge(drivers[["driverId","driverRef","code"]], on="driverId", how="left")
    df = df.merge(constructors[["constructorId","constructorRef"]], on="constructorId", how="left")

    # --------------------- Constructor lineage mapping ---------------------
    df['constructorRef'] = df['constructorRef'].str.lower()

    constructor_map = {
        "lotus_f1":"alpine","renault":"alpine","alpine_f1_team":"alpine","alpine":"alpine",
        "force_india":"aston_martin","racing_point":"aston_martin","aston_martin":"aston_martin",
        "toro_rosso":"rb_f1_team","alphatauri":"rb_f1_team","rb_f1_team":"rb_f1_team",
        "sauber":"alfa_romeo","alfa_romeo":"alfa_romeo",
        "manor_marussia":"marussia","marussia":"marussia",
        "caterham":"caterham",
        "ferrari":"ferrari","red_bull":"red_bull","mercedes":"mercedes",
        "mclaren":"mclaren","williams":"williams","haas_f1_team":"haas","haas":"haas",
    }
    df['constructorRef'] = df['constructorRef'].replace(constructor_map)

    # --------------------- Circuits info ---------------------
    df = df.merge(circuits[["circuitId","circuitRef","country","alt"]], on="circuitId", how="left")

    df = df.rename(columns={
        "position":"final_position",
        "fastestLapTime":"fastest_lap_time",
        "milliseconds":"finishing_time_ms"
    })

    # --------------------- Pit Stops Aggregation ---------------------
    df_pit = pitstops.groupby(["raceId","driverId"]).agg(
        stops_count=("stop","count"),
        total_pit_ms=("milliseconds","sum"),
        avg_pit_ms=("milliseconds","mean")
    ).reset_index()

    df = df.merge(df_pit,on=["raceId","driverId"],how="left")
    df[["stops_count","total_pit_ms","avg_pit_ms"]] = df[["stops_count","total_pit_ms","avg_pit_ms"]].fillna(0)

    # --------------------- Driver & Constructor Standings ---------------------
    df = df.merge(
        driver_standings[["raceId","driverId","points","position"]]
        .rename(columns={"points":"driver_standing_points","position":"driver_standing_position"}),
        on=["raceId","driverId"],how="left"
    )

    df = df.merge(
        constructor_standings[["raceId","constructorId","points","position"]]
        .rename(columns={"points":"constructor_points","position":"constructor_position"}),
        on=["raceId","constructorId"],how="left"
    )

    # --------------------- Qualifying best lap ---------------------
    qualifying["quali_time"]=qualifying[["q3","q2","q1"]].bfill(axis=1).iloc[:,0]
    df = df.merge(
        qualifying[["raceId","driverId","position","quali_time"]].rename(columns={"position":"quali_position"}),
        on=["raceId","driverId"],how="left"
    )

    # --------------------- Lap/Street Track Features ---------------------
    lap_length_map = { # keep unchanged from your file
        "albert_park":5.303,"sepang":5.543,"bahrain":5.412,"catalunya":4.657,"istanbul":5.338,
        "monaco":3.337,"villeneuve":4.361,"magny_cours":4.411,"silverstone":5.891,
        "hockenheimring":4.575,"hungaroring":4.381,"valencia":5.419,"spa":7.004,
        "monza":5.793,"marina_bay":5.063,"fuji":4.563,"shanghai":5.451,"interlagos":4.309,
        "indianapolis":4.192,"nurburgring":5.148,"imola":4.909,"suzuka":5.807,"vegas":6.201,
        "yas_marina":5.554,"galvez":4.259,"jerez":4.428,"estoril":4.182,"okayama":3.703,
        "adelaide":3.780,"kyalami":4.529,"donington":4.023,"rodriguez":4.304,"phoenix":3.800,
        "ricard":5.842,"yeongam":5.615,"jacarepagua":5.031,"detroit":4.000,"brands_hatch":4.207,
        "zandvoort":4.259,"dijon":3.801,"dallas":3.900,"long_beach":3.167,"las_vegas":3.650,
        "jarama":3.404,"watkins_glen":5.435,"anderstorp":4.018,"mosport":3.957,
        "montjuic":3.790,"nivelles":3.720,"charade":8.055,"tremblant":4.265,"essarts":6.542,
        "lemans":13.626,"reims":8.302,"george":3.920,"zeltweg":3.200,"aintree":4.828,
        "boavista":7.400,"riverside":5.210,"avus":8.300,"monsanto":5.440,"sebring":5.200,
        "ain-diab":7.600,"pescara":25.800,"bremgarten":7.280,"pedralbes":6.316,"buddh":5.125,
        "americas":5.513,"red_bull_ring":4.318,"sochi":5.848,"baku":6.003,"portimao":4.653,
        "mugello":5.245,"jeddah":6.174,"losail":5.380,"miami":5.410 }

    df["lap_length_km"]=df["circuitRef"].map(lap_length_map)
    df["total_distance_km"]=df["lap_length_km"]*df["laps"]

    street_ids=["monaco","marina_bay","miami","jeddah","baku","vegas","las_vegas","phoenix",
                "adelaide","detroit","dallas","long_beach","valencia"]
    df["is_street"]=df["circuitRef"].isin(street_ids).astype(int)

    # --------------------- Rolling Last 10 Form ---------------------
    df["driver_standing_points"]=pd.to_numeric(df["driver_standing_points"],errors="coerce").fillna(0)
    df["constructor_points"]=pd.to_numeric(df["constructor_points"],errors="coerce").fillna(0)

    df=df.sort_values(["year","raceId","driverId","constructorRef"])

    df["driver_points_last10"]=df.groupby("driverId")["driver_standing_points"].rolling(10,min_periods=1).sum().reset_index(level=0,drop=True)
    df["constructor_points_last10"]=df.groupby("constructorRef")["constructor_points"].rolling(10,min_periods=1).sum().reset_index(level=0,drop=True)

    # --------------------- Filter usable rows ---------------------
    df = df[df["year"]>=2013]
    df["fastest_lap_time"]=df["fastest_lap_time"].apply(time_to_ms)
    df["quali_time"]=df["quali_time"].apply(time_to_ms)
    df=df[df["finishing_time_ms"].notna()]

    # List of export columns
    final_cols=[
        "year","raceId","circuitId","circuitRef","country","statusId","driverId",
        "driverRef","constructorId","code","constructorRef","grid","final_position",
        "laps","fastest_lap_time","finishing_time_ms","stops_count","total_pit_ms",
        "avg_pit_ms","driver_standing_points","driver_standing_position",
        "constructor_points","constructor_position","quali_position","quali_time",
        "alt","lap_length_km","total_distance_km","is_street",
        "driver_points_last10","constructor_points_last10"
    ]

    df_model=df[final_cols].copy()

    # 🟩 ADD TRACK CORNER/DRS FEATURES
    df_model = add_track_features(df_model)

    df_model=enforce_dtypes(df_model)
    save_processed(df_model,"f1_modeling_dataset_final")

    return df_model



#############################################################
#  Helpers
#############################################################

def time_to_ms(x):
    if isinstance(x,str) and ":" in x:
        try:
            m,s=x.split(":"); return int(m)*60000+float(s)*1000
        except: return pd.NA
    return x


def enforce_dtypes(df):
    id_cols_32=["raceId","driverId","constructorId","circuitId","year"]
    for c in id_cols_32: df[c]=pd.to_numeric(df[c],errors="coerce").astype("int32")

    int16_cols=["grid","final_position","laps","driver_standing_position",
                "constructor_position","quali_position"]
    for c in int16_cols: df[c]=pd.to_numeric(df[c],errors="coerce").fillna(0).astype("int16")

    df["is_street"]=df["is_street"].astype("int8")

    float_cols=["finishing_time_ms","fastest_lap_time","total_pit_ms","avg_pit_ms",
                "driver_standing_points","constructor_points","quali_time","alt",
                "lap_length_km","total_distance_km","driver_points_last10",
                "constructor_points_last10"]
    for c in float_cols: df[c]=pd.to_numeric(df[c],errors="coerce").astype("float64")

    for col in ["driverRef","code","constructorRef","circuitRef","country"]:
        df[col]=df[col].astype("category")

    return df


##############################
# 🟩 Track Feature Block
##############################
def add_track_features(df):
    track_features = pd.DataFrame([
        ("bahrain",10,6,3,2),("jeddah",18,4,3,2),("australia",12,6,2,2),
        ("imola",14,6,1,1),("miami",10,8,3,2),("monaco",4,12,1,0),
        ("barcelona",12,7,2,2),("canada",8,8,3,2),("austria",14,4,3,3),
        ("silverstone",18,6,2,2),("hungary",6,12,1,1),("spa",16,8,2,3),
        ("zandvoort",12,6,2,1),("monza",16,4,2,3),("singapore",6,12,3,1),
        ("suzuka",18,6,1,2),("qatar",16,6,2,2),("mexico",10,8,3,1),
        ("brazil",10,8,2,2),("abu_dhabi",10,6,3,2),("usa",12,8,2,2),
        ("las_vegas",8,6,2,3),("china",11,7,2,2),("russia",9,9,2,2),
        ("baku",8,10,2,2),("portugal",14,6,2,2),("turkey",14,6,2,2),
        ("mugello",16,6,1,2),("nurburgring",10,8,1,1),("hockenheimring",12,6,2,2),
        ("india",13,7,2,2),("korea",10,8,2,2),("sochi",8,10,2,1),
("albert_park",10,6,2,2),
("sepang",13,5,2,2),
("shanghai",11,7,2,2),
("catalunya",12,7,2,2),
("villeneuve",8,7,2,2),
("hungaroring",6,12,1,1),
("marina_bay",6,14,3,1),
("yeongam",10,8,2,2),
("buddh",13,6,2,3),
("yas_marina",10,6,2,2),
("americas",11,8,2,2),
("interlagos",9,8,2,2),
("red_bull_ring",14,4,3,2),
("rodriguez",9,9,2,1),
("ricard",13,6,2,2),
("portimao",14,6,2,2),
("istanbul",14,6,2,2),
("losail",12,6,2,2),
("vegas",8,0,0,0)

    ],columns=["circuitRef","high_speed","low_speed","drs_zones","straights"])
    df=df.merge(track_features,on="circuitRef",how="left")

    missing=df[df["high_speed"].isna()]["circuitRef"].unique()
    if len(missing)>0:
        print("\n⚠ Unmapped Circuits — Add values in track_features:")
        for c in missing: print("   -",c)

    return df
