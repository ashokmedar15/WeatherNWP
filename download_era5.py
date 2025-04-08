import cdsapi

# Hardcode your credentials (replace with your actual values)
c = cdsapi.Client(
    url='https://cds.climate.copernicus.eu/api',
    key='7f3b7825-aabb-445f-9ae7-748216bcc4ad'
)

c.retrieve(
    'reanalysis-era5-single-levels',
    {
        'product_type': 'reanalysis',
        'format': 'netcdf',
        'variable': ['2m_temperature', 'relative_humidity', 'mean_sea_level_pressure'],
        'year': '2023',
        'month': ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12'],
        'day': [str(i).zfill(2) for i in range(1, 32)],
        'time': [f'{h:02d}:00' for h in range(24)],
        'area': [24, 68, 20, 74],  # Gujarat: N, W, S, E
    },
    'era5_gujarat_2023.nc'
)
print("Download requested! Check CDS dashboard for status.")