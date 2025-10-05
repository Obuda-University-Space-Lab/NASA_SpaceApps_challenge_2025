
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pykrige.ok import OrdinaryKriging
import matplotlib.pyplot as plt
from PIL import Image
import folium
from folium.raster_layers import ImageOverlay





def krige_data(df, gridx,gridy, bounds):
    latitude = df['latitude'].values
    longitude = df['longitude'].values
    aqi_value = df['predicted_fire'].values
    
    num_points = len(df)

    lat_min, lon_min = bounds[0]
    lat_max, lon_max = bounds[1]

    gridx = np.linspace(lon_min, lon_max, num_points)  # longitude axis (x)
    gridy = np.linspace(lat_min, lat_max, num_points)  # latitude axis (y)





    print(aqi_value)
    # Perform Ordinary Kriging using the spherical variogram model
    OK = OrdinaryKriging(longitude, 
                         latitude, 
                         aqi_value, 
                         variogram_model='spherical', 
                         coordinates_type='geographic', 
                         verbose=True, 
                         enable_plotting=True)
    z_interp, ss = OK.execute('grid', gridx, gridy)

    # Create a figure and axis
    fig, ax = plt.subplots(figsize=(8, 8))

    # Plot the interpolation results
    cax = ax.imshow(z_interp, 
                    extent=[gridx.min(), 
                            gridx.max(), 
                            gridy.min(), 
                            gridy.max()], 
                    origin='lower', 
                    cmap='magma', 
                    alpha=1)

    # Remove axes
    ax.axis('off')

    # Save the image
    fig.savefig('kriging_interpolation.png', bbox_inches='tight', pad_inches=0, transparent=True)
    plt.close(fig)

def image_overlay(gridx,gridy, bounds):
    # Define the bounds where the image will be placed
    #bounds = [[gridy.min(), gridx.min()], [gridy.max(), gridx.max()]]
    print("image_overlay: "+str(bounds))
    # Add the image overlay
    image_overlay = ImageOverlay(
        image='kriging_interpolation.png',
        bounds=bounds,
        opacity=.7,
        interactive=True,
        cross_origin=False,
        zindex=1,
    )

    return image_overlay