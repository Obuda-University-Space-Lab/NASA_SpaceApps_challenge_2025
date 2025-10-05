import pandas as pd
import meteo_data
import kriging
import datetime
import numpy as np
import joblib


def logic_func(latitude,longitude,date_to_predict,bounds):
    start_interval = date_to_predict-datetime.timedelta(days=31)
    end_interval = date_to_predict-datetime.timedelta(days=1)
    data_df= meteo_data.meteo_data_extract(latitude,longitude,start_interval,end_interval, bounds)
    model = joblib.load('./playground/model.pkl')
    dfs = {loc_id: group for loc_id, group in data_df.groupby("location_id")}
    out_df = pd.DataFrame(columns=['predicted_fire','latitude','longitude'])
    for loc_id, subdf in dfs.items():
        prediction = model.predict(subdf)
        out_data= {'predicted_fire':[prediction],'latitude':[subdf['latitude'].iloc[0]],'longitude':[subdf['longitude'].iloc[0]]}
        out_data_df = pd.DataFrame(out_data)
        out_df = pd.concat([out_df, out_data_df], ignore_index=True)

    out_df = data_df[['predicted_fire','latitude','longitude']]
    latitude = bounds[0]
    longitude = bounds[1]
    gridx = np.linspace(min(longitude), max(longitude), 100)
    gridy = np.linspace(min(latitude), max(latitude), 100)
    kriging.krige_data(out_df, gridx,gridy)
    img = kriging.image_overlay(gridx,gridy)
    return img

