#%%
from util import load_json

locations = load_json('../vast3/public/location.json')

from math import radians, sin, cos, sqrt, atan2

def calculate_distance(lat1, lon1, lat2, lon2):
    # The radius of the Earth in kilometers
    R = 6371.0

    lat1_rad = radians(lat1)
    lon1_rad = radians(lon1)
    lat2_rad = radians(lat2)
    lon2_rad = radians(lon2)

    dlon = lon2_rad - lon1_rad
    dlat = lat2_rad - lat1_rad

    a = sin(dlat / 2)**2 + cos(lat1_rad) * cos(lat2_rad) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))

    distance = R * c
    return distance

# Coordinates of Abila Hospital
abila_hospital_lat = 24.88097346
abila_hospital_lon = 36.055443485

# Define the radius in kilometers
radius_km = 1.0

locations_around_abila_hospital = []

for location in locations:
    lat1, lon1 = location["point_1"]
    distance = calculate_distance(abila_hospital_lat, abila_hospital_lon, lat1, lon1)
    if distance <= radius_km:
        locations_around_abila_hospital.append(location)

    lat2, lon2 = location["point_2"]
    distance = calculate_distance(abila_hospital_lat, abila_hospital_lon, lat2, lon2)
    if distance <= radius_km:
        locations_around_abila_hospital.append(location)

print(locations_around_abila_hospital)

# %%
