import pandas as pd
import numpy as np
from scipy.spatial import cKDTree
from geopy import distance


# Dummy function for latlon_to_xy conversion. Replace with your actual logic.
def latlon_to_xy(coords, proj):
    # Dummy implementation, replace with actual conversion logic
    return coords * proj

def latlon_to_xy(coords, proj):
    # Placeholder implementation, replace with actual conversion logic
    # For now, simply return the input coordinates assuming they are already in x, y format
    return np.array(coords, dtype=float)
def xy_to_latlon(coords, proj):
    # Placeholder implementation, replace with actual conversion logic
    # For now, simply return the input coordinates assuming they are already in lat, lon format
    return np.array(coords, dtype=float)


# Function to read data from a CSV file and convert to x, y coordinates
def read_data(file_path, type_label, proj):
    df = pd.read_csv(file_path)
    # Check the correct column names (Latitude and Longitude) in your CSV and replace 'Y' and 'X' if needed
    positions = latlon_to_xy(df[['Y', 'X']].values, proj)  # Replace 'Y' and 'X' with actual column names
    tags = np.full(len(positions), type_label)
    return np.column_stack((positions, tags))

# Assuming 'proj' is a projection conversion parameter or function
proj = 1  # Replace with actual projection conversion

# Read and tag the location data
lamp_posts = read_data('bala.csv', 'lamp post', proj)
bus_stops = read_data('/Users/balajikirubakaran/Desktop/work/Brnobusstops.csv', 'bus stop', proj)
parkings = read_data('/Users/balajikirubakaran/Desktop/work/Parking.csv', 'parking', proj)
# Uncomment the next line if you have a 'monuments.csv' file with correct column names
monuments = read_data('/Users/balajikirubakaran/Desktop/work/monuments.csv', 'monument', proj)

# Combine all location data
all_locations = np.vstack((lamp_posts, bus_stops, parkings))

# Construct a KDTree for efficient nearest neighbor search
location_tree = cKDTree(all_locations[:, :2].astype(float))

# Initialize a list to store the assigned location for each UAV
assigned_locations_for_uavs = []

# Best UAV positions provided as a list of [latitude, longitude]
best_uav_positions = [(49.19740952716811, 16.608235745795557),
   (49.19476724544848, 16.607351043170343), 
   (49.197120424076914, 16.60999171160904),
     (49.19568475682233, 16.60495279752413),
       (49.19554737994032, 16.606359675999013),
         (49.19651802895763, 16.60475553524334), 
         (49.19573802313294, 16.60592782446027), 
         (49.19434196554222, 16.604663718163845)]




 




                     





# Convert best UAV positions to x, y coordinates
best_uav_positions_xy = latlon_to_xy(np.array(best_uav_positions), proj)

# Initialize an array to keep track of which locations have been assigned
assigned_locations = np.zeros(len(all_locations), dtype=bool)

# Assign the nearest unassigned location to each UAV
for uav_pos_xy in best_uav_positions_xy:
    # Compute distances to all unassigned locations
    distances, indices = location_tree.query(uav_pos_xy, k=len(all_locations))
    for index in indices:
        if not assigned_locations[index]:
            # Mark this location as assigned
            assigned_locations[index] = True
            # Append to the list with coordinates and type
            assigned_locations_for_uavs.append(all_locations[index])
            break

# Print assigned locations for each UAV
for assigned_location in assigned_locations_for_uavs:
    coords, loc_type = assigned_location[:2], assigned_location[2]
    print(f"Coordinates: {coords}, Location Type: {loc_type}")

# Function to check if a location is within the specified radius
def is_within_radius(center_lat, center_lon, target_lat, target_lon, radius):
    center = (center_lat, center_lon)
    target = (target_lat, target_lon)
    return distance.distance(center, target).m <= radius

center_latitude = 49.1954583
center_longitude = 16.6073194
radius_in_meters = 500

# Initialize an array to keep track of which locations have been assigned
assigned_locations = np.zeros(len(all_locations), dtype=bool)

# Initialize a list to store the assigned location for each UAV
assigned_locations_for_uavs = []

# Assign the nearest unassigned location within the radius to each UAV
for uav_pos_xy in best_uav_positions_xy:
    distances, indices = location_tree.query(uav_pos_xy, k=len(all_locations))
    for index in indices:
        if not assigned_locations[index]:
            coords = all_locations[index][:2]
            lat, lon = xy_to_latlon(coords, proj)

            # Check if the location is within the specified radius
            if is_within_radius(center_latitude, center_longitude, lat, lon, radius_in_meters):
                assigned_locations[index] = True
                assigned_locations_for_uavs.append(all_locations[index])
                print(f"UAV assigned to Coordinates: {coords}, Location Type: {all_locations[index][2]}")
                break

# Final assigned locations for each UAV
print("\nFinal assigned locations for each UAV:")
for assigned_location in assigned_locations_for_uavs:
    coords, loc_type = assigned_location[:2], assigned_location[2]
    print(f"Coordinates: {coords}, Location Type: {loc_type}")





