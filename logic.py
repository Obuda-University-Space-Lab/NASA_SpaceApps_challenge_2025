import pandas as pd
import meteo_data
import kriging
import datetime
import numpy as np


def logic_func(latitude,longitude,date_to_predict,bounds):
    start_interval = date_to_predict-datetime.timedelta(days=31)
    end_interval = date_to_predict-datetime.timedelta(days=1)
    data_df= meteo_data.meteo_data_extract(latitude,longitude,start_interval,end_interval, bounds)




    df = data_df[['predicted_fire','latitude','longitude']]
    latitude = bounds[0]
    longitude = bounds[1]
    gridx = np.linspace(min(longitude), max(longitude), 100)
    gridy = np.linspace(min(latitude), max(latitude), 100)
    kriging.krige_data(df, gridx,gridy)
    img = kriging.image_overlay(gridx,gridy)
    return img

