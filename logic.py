import pandas as pd
import meteo_data
import kriging
import datetime
import numpy as np
import joblib
import random
from typing import List


def is_within_bounds(lat: float, lon: float, bounds: List[List[float]]) -> bool:
    """Check if (lat, lon) is inside rectangular bounds [[lat_min, lon_min], [lat_max, lon_max]]."""
    return (bounds[0][0] <= lat <= bounds[1][0]) and (bounds[0][1] <= lon <= bounds[1][1])


def logic_func(latitude,longitude,date_to_predict,bounds):
    start_interval = date_to_predict-datetime.timedelta(days=31)
    end_interval = date_to_predict-datetime.timedelta(days=1)
    #data_df= meteo_data.meteo_data_extract(latitude,longitude,start_interval,end_interval, bounds)
    #model = joblib.load('./playground/model.pkl')
    #dfs = {loc_id: group for loc_id, group in data_df.groupby("location_id")}

    out_df = pd.DataFrame(columns=['predicted_fire','latitude','longitude'])
    #for loc_id, subdf in dfs.items():
    #    prediction = model.predict(subdf)
    #    out_data= {'predicted_fire':[prediction],'latitude':[subdf['latitude'].iloc[0]],'longitude':[subdf['longitude'].iloc[0]]}
    #    out_data_df = pd.DataFrame(out_data)
    #    out_df = pd.concat([out_df, out_data_df], ignore_index=True)

    mockdf=pd.read_csv("/Users/lszlpotyondi/PythonProjects/NASA_SpaceApps_challenge_2025/weather_output/all_locations_weather.csv")
    loc_id=-1
    for idx, row in mockdf.iterrows():
        if int(row['location_id'])!=loc_id:
            loc_id=int(row['location_id'])
            within_bounds= is_within_bounds(row['latitude'],row['longitude'],bounds)
            if within_bounds:
                prediction= random.random()
                out_data= {'predicted_fire':[prediction],'latitude':[row['latitude']],'longitude':[row['longitude']]}
                out_data_df = pd.DataFrame(out_data)
                out_df = pd.concat([out_df, out_data_df], ignore_index=True)


    #for loc_id, subdf in dfs.items():
    #    prediction= random.random()
    #    out_data= {'predicted_fire':[prediction],'latitude':[subdf['latitude'].iloc[0]],'longitude':[subdf['longitude'].iloc[0]]}
    #    out_data_df = pd.DataFrame(out_data)
    #    out_df = pd.concat([out_df, out_data_df], ignore_index=True)


    out_df = out_df[['predicted_fire','latitude','longitude']]
    latitude = bounds[0]
    longitude = bounds[1]
    gridx = np.linspace(min(longitude), max(longitude), 100)
    gridy = np.linspace(min(latitude), max(latitude), 100)
    kriging.krige_data(out_df, gridx,gridy,bounds)
    img = kriging.image_overlay(gridx,gridy, bounds)
    return img, out_df

