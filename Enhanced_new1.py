import folium
import pandas as pd
import random
import math
import matplotlib.pyplot as plt
import numpy as np
import pickle
import geopandas as gpd
from geopy import distance
from geopy import Point
import scipy.stats as scipy_stats


import pyproj
import datetime
import scipy.stats as scipy_stats
from folium.plugins import HeatMap
from shapely.geometry import Point
from scipy.spatial.distance import cdist
from deap import base, creator, tools, algorithms
from pyproj import Transformer
from scipy.stats import nbinom


# Load the geojson file
gdf = gpd.read_file('export.geojson')
m = folium.Map()
folium.GeoJson(gdf).add_to(m)


def generate_random_points_in_square(center_lat, center_lon, num_points, side_length):
    half_side = side_length / 2
    points = []

    delta_lat = distance.distance(kilometers=half_side / 1000).destination((center_lat, center_lon), 0).latitude - center_lat
    delta_lon = distance.distance(kilometers=half_side / 1000).destination((center_lat, center_lon), 90).longitude - center_lon

    for _ in range(num_points):
        random_lat = random.uniform(center_lat - delta_lat, center_lat + delta_lat)
        random_lon = random.uniform(center_lon - delta_lon, center_lon + delta_lon)
        points.append([random_lat, random_lon])

    return points


# Define parameters and system model
N = 5000
p = 0.5 # This is equivalent to n_users_type1 / n_users
n_users_type1 = int(N * p)
n_users_type2 = int(N * (1 - p))
n_uavs = 8
bs_frequency = 3.5e9  # 3.5 GHz
bandwidth = 100e6  # 100 MHz


center_latitude = 49.1954583
center_longitude = 16.6073194
side_length_in_meters = 500
half_side = side_length_in_meters / 2

delta_lat = distance.distance(kilometers=half_side / 1000).destination((center_latitude, center_longitude), 0).latitude - center_latitude
delta_lon = distance.distance(kilometers=half_side / 1000).destination((center_latitude, center_longitude), 90).longitude - center_longitude




with open('random_points.pkl', 'rb') as file:
     random_points = pickle.load(file)

with open('random_points1.pkl', 'rb') as file:
     random_points1 = pickle.load(file)

# Define the projection you want to use (e.g. Mercator)
proj = pyproj.Transformer.from_crs("EPSG:4326", "EPSG:3857")

BS = Point(16.6073194, 49.1954583)

# Define the projection transformer
transformer = pyproj.Transformer.from_crs("EPSG:4326", "EPSG:3857")

bs1, bs2 = transformer.transform(BS.x, BS.y)
print(f"BS : ({bs1}, {bs2})")

# Convert each latitude and longitude point to x and y coordinates
def latlon_to_xy(points, proj):
    x_values = []
    y_values = []
    for lat, lon in points:
        x, y = proj.transform(lon, lat)
        x_values.append(x)
        y_values.append(y)

    # Create a NumPy array with x and y values in the rows representation
    xy_values = np.array(list(zip(x_values, y_values)))
    return xy_values



# Define the transformation - from WGS 84 (EPSG:4326) to Web Mercator (EPSG:3857)
transformer = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)

# List of UAV positions (latitude, longitude)
uav_positions = [
    (49.195309, 16.608118),
    (49.195146, 16.607497),
    (49.195098, 16.608593),
    (49.195554, 16.607866),
    (49.195473, 16.608744),
    (49.194944, 16.608749),
    (49.194727, 16.608333),
    (49.195457, 16.608225)
    #(49.195297, 16.607258),
    #(49.195404, 16.608582)
    #(49.195061, 16.608418),
    #(49.195414, 16.607734)


]



# Transform the UAV positions and store them in a NumPy array
uav_positions_transformed = np.array([transformer.transform(lat, lon) for lat, lon in uav_positions])

# Output the transformed positions
print(uav_positions_transformed)








xy_values = latlon_to_xy(random_points, proj)
xy1_values = latlon_to_xy(random_points1, proj)
#xy2_values = latlon_to_xy(random_points2, proj)

xyy2_values = uav_positions_transformed





# Vary the proportion of user types
proportions = np.linspace(0, 1)
user_counts_type1 = []
user_counts_type2 = []

for p in proportions:
    n_users_type1 = int(N * p)
    n_users_type2 = int(N * (1 - p))
    
    user_counts_type1.append(n_users_type1)
    user_counts_type2.append(n_users_type2)

# Plot the results
plt.plot(proportions, user_counts_type1, label='Type 1 Users')
plt.plot(proportions, user_counts_type2, label='Type 2 Users')
plt.xlabel('Proportion of Type 1 Users')
plt.ylabel('Number of Users')
plt.title('Impact of User Proportion on the Number of Users')
plt.legend()




# Define heights
user_height = 1.5  # meters
uav_height = 10
uav_height1 = 10
bs_height = 25 # meters

# Define transmit powers
user_tx_power = 23  # dBm
uav_tx_power = 30  # dBm
uav_tx_power1 = 30 
bs_tx_power = 46  # dBm

user_positions_type1 = xy_values
user_positions_type2 = xy1_values
uav_positions = xyy2_values


n_users_type1 = len(user_positions_type1)
n_users_type2 = len(user_positions_type2)

# Add user heights
user_positions_type1 = np.hstack([user_positions_type1, np.ones((n_users_type1, 1)) * user_height])
user_positions_type2 = np.hstack([user_positions_type2, np.ones((n_users_type2, 1)) * user_height])

# Create BS position
bs_position = np.array([bs1, bs2 , bs_height])

# Combine Type 1 and Type 2 user positions
user_positions = np.vstack([user_positions_type1, user_positions_type2])


# Define antenna gains
user_antenna_gain = 0  # dBi
uav_antenna_gain = 5  # dBi
uav_antenna_gain1 = 5 
bs_antenna_gain = 15  # dBi

shadowing_std_dev = 5 # dB
rng = np.random.default_rng(seed=None)


def pathloss_urban_macro1(distance, frequency, heights, shadowing_std_dev_los=4, shadowing_std_dev_nlos=6):
    h_ut, h_bs = heights
    fc = frequency / 1e9  # Convert to GHz
    d_2D = distance
    d_3D = np.sqrt(d_2D**2 + (h_bs - h_ut)**2)

    # Calculate breakpoint distance
    d_BP = 4 * h_bs * h_ut * fc / 3

    # LOS pathloss
    PL_LOS_1 = 28.0 + 22 * np.log10(d_2D) + 20 * np.log10(fc)
    PL_LOS_2 = 28.0 + 40 * np.log10(d_3D) + 20 * np.log10(fc) - 9 * np.log10(d_BP**2 + (h_bs - h_ut)**2)
    PL_LOS = np.where(d_2D <= d_BP, PL_LOS_1, PL_LOS_2)
    
    # NLOS pathloss
    PL_NLOS_tmp = 13.54 + 39.08 * np.log10(d_3D) + 20 * np.log10(fc) - 0.6 * (h_ut - 1.5)
    PL_NLOS = np.maximum(PL_LOS, PL_NLOS_tmp)

    # Calculate LOS probability
    p_LOS = np.minimum(18 / d_2D, 1) * (1 - np.exp(-d_2D / 63)) + np.exp(-d_2D / 63)

    # Add shadowing for LOS and NLOS
    shadowing_los = np.random.normal(0, shadowing_std_dev_los, d_2D.shape)
    shadowing_nlos = np.random.normal(0, shadowing_std_dev_nlos, d_2D.shape)
    
    # Combine LOS and NLOS pathloss values based on LOS probability
    PL = np.where(np.random.rand(*d_2D.shape) <= p_LOS, PL_LOS, PL_NLOS)
    shadowing = np.where(np.random.rand(*d_2D.shape) <= p_LOS, shadowing_los, shadowing_nlos)
    PL += shadowing

    return PL

def pathloss_urban_macro(distance, frequency, heights, shadowing_std_dev_los=4, shadowing_std_dev_nlos=6):
    h_ut, h_bs = heights
    fc = frequency / 1e9  # Convert to GHz
    d_2D = distance
    d_3D = np.sqrt(d_2D**2 + (h_bs - h_ut)**2)

    # Calculate breakpoint distance
    d_BP = 4 * h_bs * h_ut * fc / 3

    # LOS pathloss
    PL_LOS_1 = 28.0 + 22 * np.log10(d_2D) + 20 * np.log10(fc)
    PL_LOS_2 = 28.0 + 40 * np.log10(d_3D) + 20 * np.log10(fc) - 9 * np.log10(d_BP**2 + (h_bs - h_ut)**2)
    PL_LOS = np.where(d_2D <= d_BP, PL_LOS_1, PL_LOS_2)
    
    # NLOS pathloss
    PL_NLOS_tmp = 13.54 + 39.08 * np.log10(d_3D) + 20 * np.log10(fc) - 0.6 * (h_ut - 1.5)
    PL_NLOS = np.maximum(PL_LOS, PL_NLOS_tmp)

    # Calculate LOS probability
    p_LOS = np.minimum(18 / d_2D, 1) * (1 - np.exp(-d_2D / 63)) + np.exp(-d_2D / 63)

    # Add shadowing for LOS and NLOS
    shadowing_los = np.random.normal(0, shadowing_std_dev_los, d_2D.shape)
    shadowing_nlos = np.random.normal(0, shadowing_std_dev_nlos, d_2D.shape)
    
    # Generate a random array to select LOS or NLOS based on probability
    rand_arr = np.random.rand(*d_2D.shape)

    # Select LOS or NLOS pathloss based on probability and apply respective shadowing
    PL = np.where(rand_arr <= p_LOS, PL_LOS + shadowing_los, PL_NLOS + shadowing_nlos)

    return PL





def pathloss_urban_macrobackhaul(distance, frequency, heights, shadowing_std_dev_nlos=6):
    h_uav, h_bs = heights
    fc = frequency / 1e9  # Convert to GHz
    d_2D = distance
    d_3D = np.sqrt(d_2D**2 + (h_bs - h_uav)**2)

    # NLOS pathloss
    PL_NLOS = 13.54 + 39.08 * np.log10(d_3D) + 20 * np.log10(fc) - 0.6 * (h_uav - 1.5)

    # Generate shadow fading for NLOS
    shadowing_nlos = np.random.normal(0, shadowing_std_dev_nlos, d_2D.shape)

    # Add shadow fading to NLOS path loss
    PL = PL_NLOS + shadowing_nlos

    return PL









def pathloss_urban_micro1(distance, frequency, heights):
    h_ut, h_bs = heights
    fc = frequency / 1e9  # Convert to GHz
    d_2D = distance
    d_3D = np.sqrt(d_2D**2 + (h_bs - h_ut)**2)

    # Calculate breakpoint distance
    d_BP = 4 * h_bs * h_ut * fc / 3

    # UMi-LOS pathloss model
    PL_UMi_LOS_1 = 32.4 + 21 * np.log10(d_2D) + 20 * np.log10(fc)
    PL_UMi_LOS_2 = 32.4 + 40 * np.log10(d_3D) + 20 * np.log10(fc) - 9.5 * np.log10(d_BP**2 + (h_bs - h_ut)**2)
    PL_UMi_LOS = np.where(d_2D <= d_BP, PL_UMi_LOS_1, PL_UMi_LOS_2)
    shadow_fading_LOS = np.random.normal(0, 4, d_2D.shape)

    # UMi-NLOS pathloss model
    PL_UMi_NLOS = 35.3 * np.log10(d_3D) + 22.4 + 21.3 * np.log10(fc) - 0.3 * (h_ut - 1.5)
    shadow_fading_NLOS = np.random.normal(0, 7.82, d_2D.shape)

    # Calculate LOS probability
    p_LOS = np.minimum(18 / d_2D, 1) * (1 - np.exp(-d_2D / 36)) + np.exp(-d_2D / 36)

    # Combine LOS and NLOS pathloss values based on LOS probability
    PL = np.where(np.random.rand(*d_2D.shape) <= p_LOS, PL_UMi_LOS + shadow_fading_LOS, PL_UMi_NLOS + shadow_fading_NLOS)

    return PL


def pathloss_urban_micro(distance, frequency, heights):
    h_ut, h_bs = heights
    fc = frequency / 1e9  # Convert to GHz
    d_2D = distance
    d_3D = np.sqrt(d_2D**2 + (h_bs - h_ut)**2)

    # Calculate breakpoint distance
    d_BP = 4 * h_bs * h_ut * fc / 3

    # UMi-LOS pathloss model
    PL_UMi_LOS_1 = 32.4 + 20 * np.log10(d_2D) + 20 * np.log10(fc)
    PL_UMi_LOS_2 = 32.4 + 40 * np.log10(d_3D) + 20 * np.log10(fc) - 9.5 * np.log10(d_BP**2 + (h_bs - h_ut)**2)
    PL_UMi_LOS = np.where(d_2D <= d_BP, PL_UMi_LOS_1, PL_UMi_LOS_2)
    shadow_fading_LOS = np.random.normal(0, 4, d_2D.shape)

    # UMi-NLOS pathloss model
    PL_UMi_NLOS = 22.4 + 35.3 * np.log10(d_3D) + 21.3 * np.log10(fc) - 0.3 * (h_ut - 1.5)
    shadow_fading_NLOS = np.random.normal(0, 7.82, d_2D.shape)

    # Calculate LOS probability
    p_LOS = np.minimum(18 / d_2D, 1) * (1 - np.exp(-d_2D / 36)) + np.exp(-d_2D / 36)

    # Generate random numbers for comparison
    random_numbers = np.random.rand(*d_2D.shape)

    # Select LOS or NLOS pathloss values based on LOS probability
    PL = np.where(random_numbers <= p_LOS, PL_UMi_LOS + shadow_fading_LOS, PL_UMi_NLOS + shadow_fading_NLOS)

    return PL



def received_power(distance, frequency, heights, tx_power, pathloss_func, tx_antenna_gain, rx_antenna_gain):
    path_loss = pathloss_func(distance, frequency, heights)
    received_power = tx_power - path_loss + tx_antenna_gain + rx_antenna_gain
    return received_power

user_bs_distance = cdist(user_positions[:, :2], bs_position[:2][np.newaxis, :])
user_uav_distances = cdist(user_positions[:, :2], uav_positions[:, :2])

user_bs_pathloss = pathloss_urban_macro(user_bs_distance, bs_frequency, (user_height, bs_height)).flatten()
user_uav_pathloss = pathloss_urban_micro(user_uav_distances, bs_frequency, (user_height, uav_height))

uav_bs_pathloss = pathloss_urban_macrobackhaul(cdist(uav_positions[:, :2], bs_position[:2][np.newaxis, :]),bs_frequency, (uav_height, bs_height)).flatten()

user_uav_received_power = received_power(user_uav_distances, bs_frequency, (user_height, uav_height), user_tx_power, pathloss_urban_micro, user_antenna_gain, uav_antenna_gain)
user_bs_received_power = received_power(user_bs_distance, bs_frequency, (user_height, bs_height), user_tx_power, pathloss_urban_macro, user_antenna_gain, bs_antenna_gain).flatten()



uav_bs_received_power = received_power(cdist(uav_positions[:, :2], bs_position[:2][np.newaxis, :]), bs_frequency, (uav_height, bs_height), uav_tx_power, pathloss_urban_macrobackhaul, uav_antenna_gain, bs_antenna_gain).flatten()




noise_figure = 5
noise_power = -174 + 10 * np.log10(bandwidth) + noise_figure

snr_users_bs = user_bs_received_power - noise_power
user_uav_snr = user_uav_received_power - noise_power

uav_bs_snr = uav_bs_received_power - noise_power

def plot_cdf(data, label, ax, linestyle='-'):
    sorted_data = np.sort(data)
    cdf = np.arange(sorted_data.size) / sorted_data.size
    ax.plot(sorted_data, cdf, label=label, linestyle=linestyle)



# Plot 1: BS to user SNR CDF curve for type1 and type2 user
fig1 = plt.figure()
ax1 = fig1.add_subplot()
plot_cdf(snr_users_bs[:n_users_type1], 'SNR RT Users - BS', ax1)
plot_cdf(snr_users_bs[n_users_type1:], 'SNR BG Users - BS', ax1)
ax1.set_xlabel('SNR (dB)')
ax1.set_ylabel('CDF')
ax1.legend()
ax1.grid()
ax1.set_title('SNR Distribution: BS to Users')
#fig1.savefig('SNR Distribution: BS to Users.png', dpi=300)
plt.show()

# Plot 2: Each UAV to user SNR CDF curve for type1 and type2
fig2 = plt.figure()
ax2 = fig2.add_subplot()
for uav_idx in range(n_uavs):
    plot_cdf(user_uav_snr[:n_users_type1, uav_idx], f'SNR RT Users - FBS {uav_idx + 1}', ax2, linestyle='-')
    plot_cdf(user_uav_snr[n_users_type1:, uav_idx], f'SNR BG Users - FBS {uav_idx + 1}', ax2, linestyle='--')
ax2.set_xlabel('SNR (dB)')
ax2.set_ylabel('CDF')
ax2.legend()
ax2.grid()
ax2.set_title('SNR Distribution: FBSs to Users')
#fig2.savefig('SNR Distribution: UAVs to Users.png', dpi=300)
plt.show()

# Plot 3: BS to all UAVs
fig3 = plt.figure()
ax3 = fig3.add_subplot()
plot_cdf(uav_bs_snr, 'SNR FBSs - BS', ax3)
ax3.set_xlabel('SNR (dB)')
ax3.set_ylabel('CDF')
ax3.grid()
ax3.set_title('SNR Distribution: BS to FBSs')
#fig3.savefig('SNR Distribution:BS to UAVs .png', dpi=300)
plt.show()




snr_type1_users_bs = snr_users_bs[:n_users_type1]
snr_type2_users_bs = snr_users_bs[n_users_type1:]

snr_linear1 = 10 ** (snr_type1_users_bs/ 10)
snr_linear2 = 10 ** (snr_type2_users_bs/ 10)

snr_type1_users_uav = user_uav_snr[:n_users_type1, :]
snr_type2_users_uav = user_uav_snr[n_users_type1:, :]

snr_linear3 = 10 ** (snr_type1_users_uav/ 10)
snr_linear4 = 10 ** (snr_type2_users_uav/ 10)



snr_uavs_bs = uav_bs_snr
snr_linear7 = 10 ** (snr_uavs_bs/ 10)

user_positions_xy = user_positions[:, :2]  # Extract only x and y positions

user_positions_type1_xy = user_positions_xy[:n_users_type1]
user_positions_type2_xy = user_positions_xy[n_users_type1:]



N1 = user_positions_type1_xy
N2 = user_positions_type2_xy


# Define SNR values for each user
SNR1 = snr_linear1
SNR2 = snr_linear2
SNR_drones = snr_linear3
SNR_UAV_BS =  snr_linear7



# define the total number of users and number of type 1 users
N = len(N1) + len(N2)
Np = len(N1)


# Example coordinates for users, drones, and the base station
user_coords = user_positions_type1_xy
user_coords1= user_positions_type2_xy
drone_coords =  uav_positions
bs_coord = bs1,bs2,25
user_coords = np.hstack((user_positions_type1_xy, np.full((user_positions_type1_xy.shape[0], 1), 1.5)))
user_coords1 = np.hstack((user_positions_type2_xy, np.full((user_positions_type2_xy.shape[0], 1), 1.5)))
drone_coords = np.hstack((uav_positions, np.full((uav_positions.shape[0], 1), 10)))

# Calculate the distance between users and the BS
dist_BS = np.linalg.norm(user_coords - bs_coord, axis=1)

dist_UAV_BS = np.linalg.norm(drone_coords - bs_coord, axis=1)

# Calculate the distance between users and drones
dist_drones = np.linalg.norm(user_coords[:, np.newaxis, :] - drone_coords, axis=2)
dist_drones1 = np.linalg.norm(user_coords1[:, np.newaxis, :] - drone_coords, axis=2)


def truncated_negbinom(r, u, a, b):
    x = np.arange(a, b+1)
    pmf_values = nbinom.pmf(x, r, u)
    pmf_values /= pmf_values.sum()  # normalize
    return np.random.choice(x, p=pmf_values)

# Maximum supported SNR in dB
max_snr_dB = 19.6
# Convert max_snr_dB to the linear scale
max_snr = 10 ** (max_snr_dB / 10)
# Negative binomial distribution parameters
r = 10  # Number of successes (assumption, adjust as needed)
u = 0.01  # Probability of success (assumption, adjust as needed)
# Truncated negative binomial parameters
a, b = 1, 600
total_slots = int(36/0.001)
slot_duration = 0.001
observation_time = 1 * 6 * 6
B_total = 100e6

average_data_rate1 = np.zeros(Np)
average_data_rate2 = np.zeros(N - Np)
# Demand, met demand, remaining bits, and average rate for type 1 and type 2 users
D1 = np.zeros(Np)
D2 = np.zeros(N - Np)
met_demand1 = np.zeros(Np)
met_demand2 = np.zeros(N - Np)
demand1 = np.zeros(Np)
demand2 = np.zeros(N - Np)
remaining_bits1 = np.zeros(Np)
remaining_bits2 = np.zeros(N - Np)
R_avg1 = np.zeros(Np)
R_avg2 = np.zeros(N - Np)
sum_data_rate1 = np.zeros(Np)
sum_data_rate2 = np.zeros(N - Np)
count_data_rate1 = np.zeros(Np)
count_data_rate2 = np.zeros(N - Np)
active_user_counts_type1 = []
active_user_counts_type2 = []
uav_data_accumulated = np.zeros(n_uavs)
half_slot_duration = slot_duration / 2

r, u = 10, 0.01
a, b = 1, 600
max_snr_dB = 19.6
max_snr = 10 ** (max_snr_dB / 10)
B_total = 100e6
mean, sigma = 4, 0.7  # Parameters for Type 2 users


for t in range(1, total_slots + 1):
    # Generate the number of active users using the negative binomial distribution
    X1 = truncated_negbinom(r, u, a, b)
    X2 = truncated_negbinom(r, u, a, b)

    # Randomly choose active user indices
    active_type1_indices = np.random.choice(Np, X1, replace=False)
    active_type2_indices = np.random.choice(N - Np, X2, replace=False)

    active_user_counts_type1.append(X1)
    active_user_counts_type2.append(X2)

    drone_association = np.zeros(len(active_type1_indices), dtype=bool)

    for idx, user_index in enumerate(active_type1_indices):
        user_drone_snr = SNR_drones[user_index]  # SNR values from user to drones
        user_bs_snr = SNR1[user_index]  # SNR from user to base station
        drone_bs_snr = SNR_UAV_BS  # SNR from drone to base station

       # Find the bottleneck SNR for the UE-UAV-BS path
        if isinstance(drone_bs_snr, np.ndarray):
        # If drone_bs_snr is an array, find the bottleneck SNR for each drone
            bottleneck_snr_per_drone = np.minimum(user_drone_snr, drone_bs_snr)
        # Find the best SNR across all drones
            best_uav_snr = np.max(bottleneck_snr_per_drone)
        else:
        # If drone_bs_snr is a single value, it's the same for all drones
            best_uav_snr = np.min(np.minimum(user_drone_snr, drone_bs_snr))

    # Determine if the user should be associated with a drone or the base station
        drone_association[idx] = best_uav_snr > user_bs_snr



    # Bandwidth and resource allocation
    B_total = 100e6  # Bandwidth is 100 MHz
    resources_per_user = B_total / (X1 + X2)
    # Calculate PF metric for each active user
    pf_metrics = np.zeros(X1 + X2)
    # Iterate over all active users for PF metric calculation
    for i, active_user_index in enumerate(np.concatenate((active_type1_indices, active_type2_indices + Np))):
        if active_user_index < Np:  # Type 1 users
           user_index = active_user_index
           shape = 1
           scale = 2000
           bits = np.random.gamma(shape, scale)+ remaining_bits1[user_index] # Bit generation for Type 1 user
           if drone_association[np.where(active_type1_indices == user_index)[0][0]]:
              snr_UAV_UE = SNR_drones[user_index]
              snr_UAV_BS = SNR_UAV_BS
            # Determine the data rate for both fronthaul and backhaul
              fronthaul_data_rate = np.max(resources_per_user * np.log2(1 + np.max(snr_UAV_UE)))
              backhaul_data_rate = np.max(resources_per_user * np.log2(1 + snr_UAV_BS))
              # Calculate PF metric considering both fronthaul and backhaul
              max_data_rate = max(fronthaul_data_rate, backhaul_data_rate)
              r_ui = (max_data_rate + remaining_bits1[user_index]) / np.maximum(1e-10, R_avg1[user_index])
        else:
              snr = SNR1[user_index]
              achievable_data_rate = resources_per_user * np.log2(1 + snr)
              r_ui = (achievable_data_rate + remaining_bits1[user_index]) / np.maximum(1e-10, R_avg1[user_index])
    else:  # Type 2 users
            user_index = active_user_index - Np
            mean = 4
            sigma = 0.7
            bits = np.random.lognormal(mean, sigma) + remaining_bits2[user_index] # Bit generation for Type 2 user
            snr = SNR2[user_index]
            achievable_data_rate = resources_per_user * np.log2(1 + snr)
            # Calculate PF metric for Type 2 user
            r_ui = (achievable_data_rate + remaining_bits2[user_index]) / np.maximum(1e-10, R_avg2[user_index])
        
    pf_metrics[i] = r_ui

    # Sort active user indices based on PF metric
    active_user_indices_sorted = np.concatenate((active_type1_indices, active_type2_indices + Np))[np.argsort(pf_metrics)[::-1]]

    # Iterate over sorted active users for transmission
    for active_user_index in active_user_indices_sorted:
        if active_user_index < Np:  # For Type 1 users
           user_index = active_user_index
           shape = 1
           scale = 2000
           bits = np.random.gamma(shape, scale)  # Bit generation for Type 1 user
           snr_UAV_UE = SNR_drones[user_index]
           snr_UAV_BS = SNR_UAV_BS
        # Assume bits variable is defined with the number of bits to transmit
         # Check if the user is associated with a drone
           if drone_association[np.where(active_type1_indices == user_index)[0][0]]:
            # Fronthaul Transmission (UE to UAV)
              fronthaul_data_rate = np.max(resources_per_user * np.log2(1 + np.max(SNR_drones[user_index])))
              transmitted_bits_fronthaul = np.minimum(bits, fronthaul_data_rate * half_slot_duration)
              remaining_bits_after_fronthaul = bits - transmitted_bits_fronthaul
               # Backhaul Transmission (UAV to BS)
              backhaul_data_rate = np.max(resources_per_user * np.log2(1 + SNR_UAV_BS))
              transmitted_bits_backhaul = np.minimum(remaining_bits_after_fronthaul, backhaul_data_rate * half_slot_duration)
              remaining_bits1[user_index] = remaining_bits_after_fronthaul - transmitted_bits_backhaul
               # Update data transmitted and remaining bits
              D1[user_index] += transmitted_bits_backhaul
              met_demand1[user_index] += transmitted_bits_backhaul
              demand1[user_index] += bits
        else:
            # Direct Transmission (UE to BS)
              direct_data_rate = resources_per_user * np.log2(1 + SNR1[user_index])
              transmitted_bits_direct = np.minimum(bits, direct_data_rate * slot_duration)
              remaining_bits1[user_index] = bits - transmitted_bits_direct
           # Update data transmitted and remaining bits
              D1[user_index] += transmitted_bits_direct
              met_demand1[user_index] += transmitted_bits_direct
              demand1[user_index] += bits

    else:  # For Type 2 users
           user_index = active_user_index - Np
        # Assume bits variable is defined with the number of bits to transmit for Type 2 users
           mean = 4
           sigma = 0.7
           bits = np.random.lognormal(mean, sigma)
           # Direct Transmission (UE to BS)
           direct_data_rate = resources_per_user * np.log2(1 + SNR2[user_index])
           transmitted_bits_direct = np.minimum(bits, direct_data_rate * slot_duration)
           remaining_bits2[user_index] = bits - transmitted_bits_direct

           # Update data transmitted and remaining bits
           D2[user_index] += transmitted_bits_direct
           met_demand2[user_index] += transmitted_bits_direct
           demand2[user_index] += bits

# Compute total demands and the amount met for both types of users
total_demand_type1 = np.sum(demand1)
total_demand_type2 = np.sum(demand2)
met_demand_type1 = np.sum(D1)
met_demand_type2 = np.sum(D2)





 
 



# Compute data rates for Type 1 and Type 2 users and network throughput
data_rate_type1_s12 = np.sum(D1) / (Np * observation_time)
data_rate_type2_s12 = np.sum(D2) / ((N - Np) * observation_time)
network_throughput_s12 = (np.sum(D1) + np.sum(D2)) / observation_time

# Calculate met demand percentages for Type 1 and Type 2 users
met_demand_percentage_type1 = (met_demand_type1 / total_demand_type1) * 100
met_demand_percentage_type2 = (met_demand_type2 / total_demand_type2) * 100

# Convert data rates and network throughput to MB/s
data_rate_type1_s12_mbps = data_rate_type1_s12 / (1_000_000)
data_rate_type2_s12_mbps = data_rate_type2_s12 / (1_000_000)
network_throughput_s12_mbps = network_throughput_s12 / (1_000_000)

# Compute individual data rates for Type 1 and Type 2 users in Mbps
individual_data_rate_type1_s12_mbps = D1 / (observation_time * 1_000_000)
individual_data_rate_type2_s12_mbps = D2 / (observation_time * 1_000_000)            

# Create user IDs for Type 1 and Type 2 users
user_ids_type1 = np.arange(1, Np + 1)
user_ids_type2 = np.arange(Np + 1, N + 1)

print("Average Data Rate for Type 1 Users: {:.8f} MB/s".format(data_rate_type1_s12_mbps))
print("Average Data Rate for Type 2 Users: {:.8f} MB/s".format(data_rate_type2_s12_mbps))
print("Network Throughput: {:.8f} MB/s".format(network_throughput_s12_mbps))
print("Met Demand for Type 1 Users: {:.8f}%".format(met_demand_percentage_type1))
print("Met Demand for Type 2 Users: {:.8f}%".format(met_demand_percentage_type2))

    # Plotting
plt.figure(figsize=(12, 8))
plt.plot(range(len(active_user_counts_type1)), active_user_counts_type1, label="Type 1 Users")
plt.plot(range(len(active_user_counts_type2)), active_user_counts_type2, label="Type 2 Users")
plt.xlabel("Time Slot")
plt.ylabel("Number of Active Users")
plt.title("Number of Active Users vs Time Slot for Type 1 and Type 2 Users")
plt.legend()
plt.show()

def plot_cdf(data, label, color):
    sorted_data = np.sort(data)
    yvals = np.arange(len(sorted_data))/float(len(sorted_data)-1)
    plt.plot(sorted_data, yvals, label=label, color=color)
    plt.xlabel('Average Data Rate (Mbps)')
    plt.ylabel('Probability')
    plt.title('CDF of Average Data Rates for Users')
    plt.legend()

#    Generate CDF plot for average data rates
plt.figure(figsize=(10, 6))
plot_cdf(individual_data_rate_type1_s12_mbps, 'Type 1 Users', 'blue')
plot_cdf(individual_data_rate_type2_s12_mbps, 'Type 2 Users', 'red')
plt.grid(True)
plt.show()

def get_cdf_data(data):
    sorted_data = np.sort(data)
    yvals = np.arange(len(sorted_data)) / float(len(sorted_data) - 1)
    return sorted_data, yvals

# Get CDF data
sorted_data_type1, yvals_type1 = get_cdf_data(individual_data_rate_type1_s12_mbps)
sorted_data_type2, yvals_type2 = get_cdf_data(individual_data_rate_type2_s12_mbps)

# Save to a file
np.savetxt('cdf_data_type1UAV12360int.csv', np.column_stack((sorted_data_type1, yvals_type1)), delimiter=',', header='Data,Probability')
np.savetxt('cdf_data_type2UAV12360int.csv', np.column_stack((sorted_data_type2, yvals_type2)), delimiter=',', header='Data,Probability')

POPULATION_SIZE = 150  # Population size
P_CROSSOVER = 0.8  # Crossover probability
P_MUTATION = 0.2  # Mutation probability
MAX_GENERATIONS = 100  # Max number of generations
HALL_OF_FAME_SIZE = 1  # Hall of fame size (Best solutions kept)

def generate_lat_long():
    return [random.uniform(center_latitude - delta_lat, center_latitude + delta_lat), 
            random.uniform(center_longitude - delta_lon, center_longitude + delta_lon)]




def simulate_network(uav_positions):
    # Maximum supported SNR in dB
    max_snr_dB = 19.6
    # Convert max_snr_dB to the linear scale
    max_snr = 10 ** (max_snr_dB / 10)
    # Negative binomial distribution parameters
    r = 10  # Number of successes (assumption, adjust as needed)
    u = 0.01  # Probability of success (assumption, adjust as needed)
    # Truncated negative binomial parameters
    a, b = 1, 600
    total_slots = int(36/0.001)
    slot_duration = 0.001
    observation_time = 1 * 6 * 6
    B_total = 100e6

    average_data_rate1 = np.zeros(Np)
    average_data_rate2 = np.zeros(N - Np)
    # Demand, met demand, remaining bits, and average rate for type 1 and type 2 users
    D1 = np.zeros(Np)
    D2 = np.zeros(N - Np)
    met_demand1 = np.zeros(Np)
    met_demand2 = np.zeros(N - Np)
    demand1 = np.zeros(Np)
    demand2 = np.zeros(N - Np)
    remaining_bits1 = np.zeros(Np)
    remaining_bits2 = np.zeros(N - Np)
    R_avg1 = np.zeros(Np)
    R_avg2 = np.zeros(N - Np)
    sum_data_rate1 = np.zeros(Np)
    sum_data_rate2 = np.zeros(N - Np)
    count_data_rate1 = np.zeros(Np)
    count_data_rate2 = np.zeros(N - Np)
    active_user_counts_type1 = []
    active_user_counts_type2 = []
    uav_data_accumulated = np.zeros(n_uavs)
    half_slot_duration = slot_duration / 2

    r, u = 10, 0.01
    a, b = 1, 600
    max_snr_dB = 19.6
    max_snr = 10 ** (max_snr_dB / 10)
    B_total = 100e6
    mean, sigma = 4, 0.7  # Parameters for Type 2 users

    for t in range(1, total_slots + 1):
    # Generate the number of active users using the negative binomial distribution
        X1 = truncated_negbinom(r, u, a, b)
        X2 = truncated_negbinom(r, u, a, b)

        # Randomly choose active user indices
        active_type1_indices = np.random.choice(Np, X1, replace=False)
        active_type2_indices = np.random.choice(N - Np, X2, replace=False)

        active_user_counts_type1.append(X1)
        active_user_counts_type2.append(X2)

        drone_association = np.zeros(len(active_type1_indices), dtype=bool)

        for idx, user_index in enumerate(active_type1_indices):
            user_drone_snr = SNR_drones[user_index]  # SNR values from user to drones
            user_bs_snr = SNR1[user_index]  # SNR from user to base station
            drone_bs_snr = SNR_UAV_BS  # SNR from drone to base station

            # Find the bottleneck SNR for the UE-UAV-BS path
            if isinstance(drone_bs_snr, np.ndarray):
           # If drone_bs_snr is an array, find the bottleneck SNR for each drone
               bottleneck_snr_per_drone = np.minimum(user_drone_snr, drone_bs_snr)
        # Find the best SNR across all drones
               best_uav_snr = np.max(bottleneck_snr_per_drone)
            else:
         # If drone_bs_snr is a single value, it's the same for all drones
                best_uav_snr = np.min(np.minimum(user_drone_snr, drone_bs_snr))

            # Determine if the user should be associated with a drone or the base station
            drone_association[idx] = best_uav_snr > user_bs_snr



       # Bandwidth and resource allocation
        B_total = 100e6  # Bandwidth is 100 MHz
        resources_per_user = B_total / (X1 + X2)
        # Calculate PF metric for each active user
        pf_metrics = np.zeros(X1 + X2)
        # Iterate over all active users for PF metric calculation
        for i, active_user_index in enumerate(np.concatenate((active_type1_indices, active_type2_indices + Np))):
            if active_user_index < Np:  # Type 1 users
               user_index = active_user_index
               shape = 1
               scale = 2000
               bits = np.random.gamma(shape, scale)+ remaining_bits1[user_index] # Bit generation for Type 1 user
               if drone_association[np.where(active_type1_indices == user_index)[0][0]]:
                  snr_UAV_UE = SNR_drones[user_index]
                  snr_UAV_BS = SNR_UAV_BS
                # Determine the data rate for both fronthaul and backhaul
                  fronthaul_data_rate = np.max(resources_per_user * np.log2(1 + np.max(snr_UAV_UE)))
                  backhaul_data_rate = np.max(resources_per_user * np.log2(1 + snr_UAV_BS))
                  # Calculate PF metric considering both fronthaul and backhaul
                  max_data_rate = max(fronthaul_data_rate, backhaul_data_rate)
                  r_ui = (max_data_rate + remaining_bits1[user_index]) / np.maximum(1e-10, R_avg1[user_index])
            else:
                  snr = SNR1[user_index]
                  achievable_data_rate = resources_per_user * np.log2(1 + snr)
                  r_ui = (achievable_data_rate + remaining_bits1[user_index]) / np.maximum(1e-10, R_avg1[user_index])
        else:  # Type 2 users
                user_index = active_user_index - Np
                mean = 4
                sigma = 0.7
                bits = np.random.lognormal(mean, sigma) + remaining_bits2[user_index] # Bit generation for Type 2 user
                snr = SNR2[user_index]
                achievable_data_rate = resources_per_user * np.log2(1 + snr)
            # Calculate PF metric for Type 2 user
                r_ui = (achievable_data_rate + remaining_bits2[user_index]) / np.maximum(1e-10, R_avg2[user_index])
        
        pf_metrics[i] = r_ui

    # Sort active user indices based on PF metric
        active_user_indices_sorted = np.concatenate((active_type1_indices, active_type2_indices + Np))[np.argsort(pf_metrics)[::-1]]

        # Iterate over sorted active users for transmission
        for active_user_index in active_user_indices_sorted:
            if active_user_index < Np:  # For Type 1 users
               user_index = active_user_index
               shape = 1
               scale = 2000
               bits = np.random.gamma(shape, scale)  # Bit generation for Type 1 user
               snr_UAV_UE = SNR_drones[user_index]
               snr_UAV_BS = SNR_UAV_BS
             # Assume bits variable is defined with the number of bits to transmit
             # Check if the user is associated with a drone
               if drone_association[np.where(active_type1_indices == user_index)[0][0]]:
                # Fronthaul Transmission (UE to UAV)
                  fronthaul_data_rate = np.max(resources_per_user * np.log2(1 + np.max(SNR_drones[user_index])))
                  transmitted_bits_fronthaul = np.minimum(bits, fronthaul_data_rate * half_slot_duration)
                  remaining_bits_after_fronthaul = bits - transmitted_bits_fronthaul
                   # Backhaul Transmission (UAV to BS)
                  backhaul_data_rate = np.max(resources_per_user * np.log2(1 + SNR_UAV_BS))
                  transmitted_bits_backhaul = np.minimum(remaining_bits_after_fronthaul, backhaul_data_rate * half_slot_duration)
                  remaining_bits1[user_index] = remaining_bits_after_fronthaul - transmitted_bits_backhaul
                   # Update data transmitted and remaining bits
                  D1[user_index] += transmitted_bits_backhaul
                  met_demand1[user_index] += transmitted_bits_backhaul
                  demand1[user_index] += bits
            else:
                # Direct Transmission (UE to BS)
                  direct_data_rate = resources_per_user * np.log2(1 + SNR1[user_index])
                  transmitted_bits_direct = np.minimum(bits, direct_data_rate * slot_duration)
                  remaining_bits1[user_index] = bits - transmitted_bits_direct
               # Update data transmitted and remaining bits
                  D1[user_index] += transmitted_bits_direct
                  met_demand1[user_index] += transmitted_bits_direct
                  demand1[user_index] += bits

        else:  # For Type 2 users
               user_index = active_user_index - Np
             # Assume bits variable is defined with the number of bits to transmit for Type 2 users
               mean = 4
               sigma = 0.7
               bits = np.random.lognormal(mean, sigma)
               # Direct Transmission (UE to BS)
               direct_data_rate = resources_per_user * np.log2(1 + SNR2[user_index])
               transmitted_bits_direct = np.minimum(bits, direct_data_rate * slot_duration)
               remaining_bits2[user_index] = bits - transmitted_bits_direct

               # Update data transmitted and remaining bits
               D2[user_index] += transmitted_bits_direct
               met_demand2[user_index] += transmitted_bits_direct
               demand2[user_index] += bits

        # Calculate the average data rate for Type 1 users
        average_data_rate_type1 = np.sum(D1) / (Np * observation_time)

        return average_data_rate_type1
    

# Fitness Function
def fitness(individual):
    uav_positions = np.array(individual).reshape(-1, 2)
    
    # Convert latitude-longitude to X-Y coordinates (if needed)
    uav_positions_xy = latlon_to_xy(uav_positions, proj)  # Replace with actual conversion logic

    # Call the simulate_network function to get the average data rate
    average_data_rate_type1 = simulate_network(uav_positions_xy)

    return average_data_rate_type1,  # Fitness value

# Genetic Algorithm Setup
creator.create("FitnessMax", base.Fitness, weights=(1.0,))  # Maximize the fitness value
creator.create("Individual", list, fitness=creator.FitnessMax)

def mutPolynomialBoundedCoordinates(individual, eta, low, up, indpb):
    for i in range(len(individual)):
        if random.random() <= indpb:
            x, y = individual[i]
            delta_1 = (x - low) / (up - low)
            delta_2 = (up - x) / (up - low)
            rand = random.random()
            if rand < 0.5:
                xy = delta_1
                delta_q = (2.0 * rand + (1.0 - 2.0 * rand) * (1.0 - xy)**(eta + 1.0))**(1.0 / (eta + 1.0)) - 1.0
            else:
                xy = delta_2
                delta_q = 1.0 - (2.0 * (1.0 - rand) + 2.0 * (rand - 0.5) * (1.0 - xy)**(eta + 1.0))**(1.0 / (eta + 1.0))
            x += delta_q * (up - low)
            individual[i] = [max(min(x, up), low), y]
    return individual,

#def mutPolynomialBoundedCoordinates(individual, eta, low, up, indpb):
    # The mutate function will be applied to each pair of coordinates in the individual
    #for i in range(len(individual)):
       # if random.random() <= indpb:
           # x, y = individual[i]
            #delta_1 = (x - low) / (up - low)
            #delta_2 = (up - x) / (up - low)
            #rand = random.random()
            #if rand < 0.5:
               # xy = delta_1
                #val = 2.0*rand + (1.0-2.0*rand)*(1.0-xy)**(eta+1.0)
                #delta_q = val**((1.0)/(eta+1)) - 1.0
            #else:
                #xy = delta_2
                #val = 2.0*(1.0-rand)+2.0*(rand-0.5)*(1.0-xy)**(eta+1.0)
                #delta_q = 1.0 - val**((1.0)/(eta+1))
            #x += delta_q*(up - low)
            #x = min(max(x, low), up)
           # individual[i] = [x, y]
    #return individual,

toolbox = base.Toolbox()
toolbox.register("attr_latlon", generate_lat_long)
toolbox.register("individual_creator", tools.initRepeat, creator.Individual, toolbox.attr_latlon, n_uavs)
toolbox.register("population_creator", tools.initRepeat, list, toolbox.individual_creator)

toolbox.register("evaluate", fitness)
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", mutPolynomialBoundedCoordinates, eta=20.0, low=center_latitude - delta_lat, up=center_latitude + delta_lat, indpb=0.2)

# Running the Genetic Algorithm
population = toolbox.population_creator(n=POPULATION_SIZE)
hall_of_fame = tools.HallOfFame(HALL_OF_FAME_SIZE)

stats = tools.Statistics(lambda ind: ind.fitness.values)
stats.register("avg", np.mean)
stats.register("std", np.std)
stats.register("min", np.min)
stats.register("max", np.max)

population, logbook = algorithms.eaSimple(population, toolbox, cxpb=P_CROSSOVER, mutpb=P_MUTATION, ngen=MAX_GENERATIONS, stats=stats, halloffame=hall_of_fame, verbose=True)

# Extracting and Printing the Best Solution
best_solution = hall_of_fame.items[0]
print(f"Best Solution (UAV Positions): {best_solution}")
print(f"Best Average Data Rate for Type 1 Users: {best_solution.fitness.values[0]} Mbps")

xy2_valuesnew= latlon_to_xy(best_solution, proj)

best_uav_positions=xy2_valuesnew











