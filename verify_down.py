import netCDF4 as nc

# Load the file
ds = nc.Dataset('C:/Users/91905/PycharmProjects/WeatherNWP/era5_gujarat_2023.nc')
print("Variables:", list(ds.variables.keys()))
print("Temperature shape:", ds.variables['t2m'].shape)
print("Sample temperature (first time, lat, lon):", ds.variables['t2m'][0, 0, 0])
ds.close()

import netCDF4 as nc

ds = nc.Dataset('C:/Users/91905/PycharmProjects/WeatherNWP/era5_gujarat_2023.nc')
print("Variables:", list(ds.variables.keys()))
for var in ds.variables:
    print(f"{var}: {ds.variables[var].long_name if hasattr(ds.variables[var], 'long_name') else 'No long name'}")
ds.close()