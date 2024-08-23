import folium
import pandas as pd
import random
import math
import matplotlib.pyplot as plt
import numpy as np
import geopandas as gpd
from geopy import distance
from geopy import Point
from pyproj import Transformer
import scipy.stats as scipy_stats
import pyproj
import datetime
import scipy.stats as scipy_stats
from folium.plugins import HeatMap
from shapely.geometry import Point
from scipy.spatial.distance import cdist
from deap import base, creator, tools
from deap import algorithms


# Load the geojson file
gdf = gpd.read_file('export.geojson')
m = folium.Map()
folium.GeoJson(gdf).add_to(m)


def generate_random_points_in_square(center_lat, center_lon, num_points, side_length):
    half_side = side_length / 2
    points = []

    delta_lat = distance.distance(kilometers=half_side / 600).destination((center_lat, center_lon), 0).latitude - center_lat
    delta_lon = distance.distance(kilometers=half_side / 600).destination((center_lat, center_lon), 90).longitude - center_lon

    for _ in range(num_points):
        random_lat = random.uniform(center_lat - delta_lat, center_lat + delta_lat)
        random_lon = random.uniform(center_lon - delta_lon, center_lon + delta_lon)
        points.append([random_lat, random_lon])

    return points


# Example usage:
center_latitude = 49.195309
center_longitude = 16.608118
side_length_in_meters = 600
half_side = side_length_in_meters / 2

delta_lat = distance.distance(kilometers=half_side / 600).destination((center_latitude, center_longitude), 0).latitude - center_latitude
delta_lon = distance.distance(kilometers=half_side / 600).destination((center_latitude, center_longitude), 90).longitude - center_longitude

# Define parameters and system model
N = 5000
p = 0.5 # This is equivalent to n_users_type1 / n_users
n_users_type1 = int(N * p)
n_users_type2 = int(N * (1 - p))
n_uavs = 1
n1_uavs = 4
bs_frequency = 3.5e9  # 3.5 GHz
bandwidth = 100e6  # 100 MHz


random_points = generate_random_points_in_square(center_latitude, center_longitude, n_users_type1, side_length_in_meters)
random_points1 = generate_random_points_in_square(center_latitude, center_longitude, n_users_type2, side_length_in_meters)
#random_points2 = generate_random_points_in_square(center_latitude, center_longitude, n_uavs, side_length_in_meters)

# Define the projection you want to use (e.g. Mercator)
proj = pyproj.Transformer.from_crs("EPSG:4326", "EPSG:3857")

BS = Point(16.608118, 49.195309)

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

xy_values = latlon_to_xy(random_points, proj)
xy1_values = latlon_to_xy(random_points1, proj)


# Define the transformation - from WGS 84 (EPSG:4326) to Web Mercator (EPSG:3857)
transformer = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)

# List of UAV positions (latitude, longitude)
uav_positions = [
    (49.195309, 16.608118)
    #(49.195146, 16.607497),
   #(49.195098, 16.608593),
    #(49.195554, 16.607866),
    #(49.195473, 16.608744)
    #(49.194944, 16.608749),
    #(49.194727, 16.608333)
]




# Transform the UAV positions and store them in a NumPy array
uav_positions_transformed = np.array([transformer.transform(lat, lon) for lat, lon in uav_positions])

# Output the transformed positions
print(uav_positions_transformed)


# Define the transformation - from WGS 84 (EPSG:4326) to Web Mercator (EPSG:3857)
transformer1 = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)

# List of UAV positions (latitude, longitude)
uav_positions1 = [
   #(49.195309, 16.608118)
    #(49.195146, 16.607497),
    (49.195098, 16.608593),
    (49.194832, 16.607895),
    #(49.195835, 16.607995),
    (49.194944, 16.608749),
    (49.194727, 16.608333)
]


# Transform the UAV positions and store them in a NumPy array
uav_positions_transformed1 = np.array([transformer1.transform(lat, lon) for lat, lon in uav_positions1])

# Output the transformed positions
print(uav_positions_transformed1)





xy_values = latlon_to_xy(random_points, proj)
xy1_values = latlon_to_xy(random_points1, proj)
xy2_values = uav_positions_transformed
xy2_values1 = uav_positions_transformed1

uav_positions = xy2_values
uav_positions1 = xy2_values1


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
bs_height = 25 # meters

# Define transmit powers
user_tx_power = 23  # dBm
uav_tx_power = 30  # dBm
bs_tx_power = 46  # dBm

user_positions_type1 = xy_values
user_positions_type2 = xy1_values


n_users_type1 = len(user_positions_type1)
n_users_type2 = len(user_positions_type2)

# Add user heights
user_positions_type1 = np.hstack([user_positions_type1, np.ones((n_users_type1, 1)) * user_height])
user_positions_type2 = np.hstack([user_positions_type2, np.ones((n_users_type2, 1)) * user_height])

# Create BS position
bs_position = np.array([bs1, bs2 , bs_height])

# Combine Type 1 and Type 2 user positions
user_positions = np.vstack([user_positions_type1, user_positions_type2])



# Parameters
POPULATION_SIZE = 100  # Population size
P_CROSSOVER = 0.7  # Crossover probability
P_MUTATION = 0.3  # Mutation probability
MAX_GENERATIONS = 500  # Max number of generations
HALL_OF_FAME_SIZE = 1  # Hall of fame size (Best solutions kept)

def generate_lat_long():
    return [random.uniform(center_latitude - delta_lat, center_latitude + delta_lat), 
            random.uniform(center_longitude - delta_lon, center_longitude + delta_lon)]

#def fitness(individual):
    #uav_positions1 = np.array(individual).reshape(-1, 2)
    #uav_positions1 = latlon_to_xy(uav_positions1, proj)  # convert to x, y
    #min_distances = np.min(cdist(user_positions_type1[:, :2], uav_positions1), axis=1)
    #return -np.sum(min_distances),  # Negative because we want to minimize distance

def fitness(individual):
    # Convert UAV positions from latitude and longitude to XY coordinates
    uav_positions = np.array(individual).reshape(-1, 2)
    uav_positions = latlon_to_xy(uav_positions, proj)

    # Calculate the minimum distances from each user to the nearest UAV
    distances = cdist(user_positions_type1[:, :2], uav_positions)
    min_distances = np.min(distances, axis=1)

    # Signal quality estimation (based on distance)
    signal_quality = -np.sum(min_distances)  # Negative, as we aim to minimize distance

    # Load balancing metric
    # Count number of users per UAV (assuming users connect to the nearest UAV)
    users_per_uav = np.argmin(distances, axis=1)
    uav_loads = np.bincount(users_per_uav, minlength=len(uav_positions))
    load_variance = np.var(uav_loads)  # We aim to minimize variance in load

    # Combine the metrics
    # Note: You may need to adjust weights (alpha, beta) based on importance of each metric
    alpha, beta = 1, 1
    combined_fitness = alpha * signal_quality + beta * load_variance

    return combined_fitness,


def mutPolynomialBoundedCoordinates(individual, eta, low, up, indpb):
    # The mutate function will be applied to each pair of coordinates in the individual
    for i in range(len(individual)):
        if random.random() <= indpb:
            x, y = individual[i]
            delta_1 = (x - low) / (up - low)
            delta_2 = (up - x) / (up - low)
            rand = random.random()
            if rand < 0.5:
                xy = delta_1
                val = 2.0*rand + (1.0-2.0*rand)*(1.0-xy)**(eta+1.0)
                delta_q = val**((1.0)/(eta+1)) - 1.0
            else:
                xy = delta_2
                val = 2.0*(1.0-rand)+2.0*(rand-0.5)*(1.0-xy)**(eta+1.0)
                delta_q = 1.0 - val**((1.0)/(eta+1))
            x += delta_q*(up - low)
            x = min(max(x, low), up)
            individual[i] = [x, y]
    return individual,




# Create types
creator.create("FitnessMin", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)

# Create toolbox
toolbox = base.Toolbox()
toolbox.register("attr_latlon", generate_lat_long)
toolbox.register("individual_creator", tools.initRepeat, creator.Individual, toolbox.attr_latlon, n1_uavs)
toolbox.register("population_creator", tools.initRepeat, list, toolbox.individual_creator)

# Genetic operators
toolbox.register("evaluate", fitness)
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("mate", tools.cxTwoPoint)


toolbox.register("mutate", mutPolynomialBoundedCoordinates, eta=20.0, low=center_latitude - delta_lat, up=center_latitude + delta_lat, indpb=0.1)


# Population
population = toolbox.population_creator(n=POPULATION_SIZE)
hall_of_fame = tools.HallOfFame(HALL_OF_FAME_SIZE)

stats = tools.Statistics(lambda ind: ind.fitness.values)
stats.register("avg", np.mean)
stats.register("std", np.std)
stats.register("min", np.min)
stats.register("max", np.max)


# Now you can use algorithms.eaSimple instead of tools.eaSimple
population, logbook = algorithms.eaSimple(population, toolbox, cxpb=P_CROSSOVER, mutpb=P_MUTATION, ngen=MAX_GENERATIONS, stats=stats, halloffame=hall_of_fame, verbose=True)




# Print best solution info
best_solution = hall_of_fame.items[0]
print(f"Best Solution: {best_solution}")
print(f"Best Fitness: {best_solution.fitness.values[0]}")

xy2_valuesnew= latlon_to_xy(best_solution, proj)

best_uav_positions=xy2_valuesnew


df = pd.read_csv("bala.csv")

for index, row in df.iterrows():
    x3 = row["X"]
    y3 = row["Y"]
    folium.CircleMarker(location=[y3, x3], radius=4, color='green', fill=True, fill_color='green').add_to(m)

for point in random_points:
    folium.CircleMarker(location=point,radius=3.5,color='red',fill=True,fill_color='red').add_to(m)

for point1 in random_points1:
    folium.CircleMarker(location=point1,radius=2.0,color='orange',fill=True,fill_color='orange').add_to(m)

#for point in points:
    #folium.Marker(location=point, popup='Random Point', icon=folium.Icon(color='blue')).add_to(m)
folium.Marker(location=[49.1954583,16.6073194],popup='Marker',icon=folium.Icon(color='blue')).add_to(m)


legend_html = """
     <div style="position: fixed; 
     bottom: 50px; left: 50px; width: auto; height: auto; 
     border:2px solid grey; z-index:9999; font-size:auto;
     ">&nbsp; <br>
     &nbsp; Users1 &nbsp; <i class="fa fa-circle" style="color:red"></i><br>
     &nbsp; Users2 &nbsp; <i class="fa fa-circle" style="color:yellow"></i><br>
     &nbsp; gNodeB &nbsp; <i class="fa fa-circle" style="color:blue"></i><br>
     &nbsp; LampPosts &nbsp; <i class="fa fa-circle" style="color:green"></i>
     </div>
     """

m.get_root().html.add_child(folium.Element(legend_html))



zoom_html = '''
<a href="javascript:void(0);" 
onclick="document.getElementById('mapid').style.height='700px'; 
document.getElementById('mapid').style.width='100%'; 
document.getElementById('mapid').style.position='fixed'; 
document.getElementById('mapid').style.zIndex='9999'; 
document.getElementById('mapid').style.top='0'; 
document.getElementById('mapid').style.left='0'; 
document.getElementById('mapid').style.right='0'; 
document.getElementById('mapid').style.bottom='0'; 
document.getElementById('mapid').style.margin='auto';">Fullscreen</a>
'''
m.get_root().html.add_child(folium.Element(zoom_html))
m.save("map.html")


# Read the lamp post data
df = pd.read_csv("bala.csv")

# Convert the lamp post positions to x, y coordinates
lamp_post_positions = latlon_to_xy(df[['Y', 'X']].values, proj)  # Assuming 'Y' is latitude and 'X' is longitude

# Initialize an array to keep track of which lamp posts have been assigned
assigned_lamp_posts = np.zeros(len(lamp_post_positions), dtype=bool)

# Initialize a list to store the assigned lamp post for each UAV
assigned_lamp_posts_for_uavs = []

# For each UAV position
for uav_position1 in best_uav_positions:
    # Compute the distances to all lamp posts
    distances = np.linalg.norm(lamp_post_positions - uav_position1, axis=1)

    # Find the index of the nearest unassigned lamp post
    unassigned_distances = distances.copy()
    unassigned_distances[assigned_lamp_posts] = np.inf  # Set the distances of assigned lamp posts to infinity
    nearest_lamp_post_index = np.argmin(unassigned_distances)

    # Mark the nearest lamp post as assigned
    assigned_lamp_posts[nearest_lamp_post_index] = True

    # Store the assigned lamp post for this UAV
    assigned_lamp_posts_for_uavs.append(lamp_post_positions[nearest_lamp_post_index])

# Now, assigned_lamp_posts_for_uavs contains the positions of the assigned lamp posts for each UAV.
print(f"Assigned Lamp Posts for UAVs: {assigned_lamp_posts_for_uavs}")


uav_positions1 = assigned_lamp_posts_for_uavs
# convert list of arrays into a numpy array
uav_positions1 = np.array(uav_positions1)







# Define antenna gains
user_antenna_gain = 0  # dBi
uav_antenna_gain = 5  # dBi
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
user_uav_distances1 = cdist(user_positions[:, :2], uav_positions1[:, :2])
user_bs_pathloss = pathloss_urban_macro(user_bs_distance, bs_frequency, (user_height, bs_height)).flatten()
user_uav_pathloss = pathloss_urban_micro(user_uav_distances, bs_frequency, (user_height, uav_height))
user_uav_pathloss1 = pathloss_urban_micro(user_uav_distances1, bs_frequency, (user_height, uav_height))
uav_bs_pathloss = pathloss_urban_macrobackhaul(cdist(uav_positions[:, :2], bs_position[:2][np.newaxis, :]),bs_frequency, (uav_height, bs_height)).flatten()


user_uav_received_power = received_power(user_uav_distances, bs_frequency, (user_height, uav_height), user_tx_power, pathloss_urban_micro, user_antenna_gain, uav_antenna_gain)

user_uav_received_power1 = received_power(user_uav_distances1, bs_frequency, (user_height, uav_height), user_tx_power, pathloss_urban_micro, user_antenna_gain, uav_antenna_gain)
user_bs_received_power = received_power(user_bs_distance, bs_frequency, (user_height, bs_height), user_tx_power, pathloss_urban_macro, user_antenna_gain, bs_antenna_gain).flatten()



uav_bs_received_power = received_power(cdist(uav_positions[:, :2], bs_position[:2][np.newaxis, :]), bs_frequency, (uav_height, bs_height), uav_tx_power, pathloss_urban_macrobackhaul, uav_antenna_gain, bs_antenna_gain).flatten()




noise_figure = 5
noise_power = -174 + 10 * np.log10(bandwidth) + noise_figure

snr_users_bs = user_bs_received_power - noise_power
user_uav_snr = user_uav_received_power - noise_power
user_uav_snr1 = user_uav_received_power1 - noise_power
uav_bs_snr = uav_bs_received_power - noise_power

def plot_cdf(data, label, ax, linestyle='-'):
    sorted_data = np.sort(data)
    cdf = np.arange(sorted_data.size) / sorted_data.size
    ax.plot(sorted_data, cdf, label=label, linestyle=linestyle)



# Plot 1: BS to user SNR CDF curve for type1 and type2 user
fig1 = plt.figure()
ax1 = fig1.add_subplot()
plot_cdf(snr_users_bs[:n_users_type1], 'SNR Type 1 Users - BS', ax1)
plot_cdf(snr_users_bs[n_users_type1:], 'SNR Type 2 Users - BS', ax1)
ax1.set_xlabel('SNR (dB)')
ax1.set_ylabel('CDF')
ax1.legend()
ax1.grid()
ax1.set_title('SNR Distribution: BS to Users')
fig1.savefig('SNR Distribution: BS to Users.png', dpi=300)

# Plot 2: Each UAV to user SNR CDF curve for type1 and type2
fig2 = plt.figure()
ax2 = fig2.add_subplot()
for uav_idx in range(n_uavs):
    plot_cdf(user_uav_snr[:n_users_type1, uav_idx], f'SNR Type 1 Users - UAV {uav_idx + 1}', ax2, linestyle='-')
    plot_cdf(user_uav_snr[n_users_type1:, uav_idx], f'SNR Type 2 Users - UAV {uav_idx + 1}', ax2, linestyle='--')
ax2.set_xlabel('SNR (dB)')
ax2.set_ylabel('CDF')
ax2.legend()
ax2.grid()
ax2.set_title('SNR Distribution: UAVs to Users')
fig2.savefig('SNR Distribution: UAVs to Users.png', dpi=300)

# Plot 3: BS to all UAVs
fig3 = plt.figure()
ax3 = fig3.add_subplot()
plot_cdf(uav_bs_snr, 'SNR UAVs - BS', ax3)
ax3.set_xlabel('SNR (dB)')
ax3.set_ylabel('CDF')
ax3.grid()
ax3.set_title('SNR Distribution: BS to UAVs')
fig3.savefig('SNR Distribution:BS to UAVs .png', dpi=300)

fig4 = plt.figure()
ax4 = fig4.add_subplot()
for uav_idx1 in range(n1_uavs):
    plot_cdf(user_uav_snr1[:n_users_type1, uav_idx1], f'SNR Type 1 Users - UAV {uav_idx1 + 1}', ax4, linestyle='-')
    plot_cdf(user_uav_snr1[n_users_type1:, uav_idx1], f'SNR Type 2 Users - UAV {uav_idx1 + 1}', ax4, linestyle='--')
ax4.set_xlabel('SNR (dB)')
ax4.set_ylabel('CDF')
ax4.legend()
ax4.grid()
ax4.set_title('SNR Distribution: UAVs to Users')
fig4.savefig('SNR Distribution: UAVs to Users.png', dpi=300)




snr_type1_users_bs = snr_users_bs[:n_users_type1]
snr_type2_users_bs = snr_users_bs[n_users_type1:]

snr_linear1 = 10 ** (snr_type1_users_bs/ 10)
snr_linear2 = 10 ** (snr_type2_users_bs/ 10)

snr_type1_users_uav = user_uav_snr[:n_users_type1, :]
snr_type2_users_uav = user_uav_snr[n_users_type1:, :]

snr_linear3 = 10 ** (snr_type1_users_uav/ 10)
snr_linear4 = 10 ** (snr_type2_users_uav/ 10)

snr_type1_users_uav1 = user_uav_snr1[:n_users_type1, :]
snr_type2_users_uav1 = user_uav_snr1[n_users_type1:, :]

snr_linear5 = 10 ** (snr_type1_users_uav1/ 10)
snr_linear6 = 10 ** (snr_type2_users_uav1/ 10)

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
SNR_drones1 = snr_linear4
SNR_drones_ext = snr_linear5
snr_drones_ext1 = snr_linear6
SNR_UAV_BS =  snr_linear7

# define the total number of users and number of type 1 users
N = len(N1) + len(N2)
Np = len(N1)


# Example coordinates for users, drones, and the base station
user_coords = user_positions_type1_xy
user_coords1= user_positions_type2_xy
drone_coords =  uav_positions
drone_coords1 =  uav_positions1
bs_coord = bs1,bs2,25
user_coords = np.hstack((user_positions_type1_xy, np.full((user_positions_type1_xy.shape[0], 1), 1.5)))
user_coords1 = np.hstack((user_positions_type2_xy, np.full((user_positions_type2_xy.shape[0], 1), 1.5)))
drone_coords = np.hstack((uav_positions, np.full((uav_positions.shape[0], 1), 10)))
drone_coords1 = np.hstack((uav_positions1, np.full((uav_positions1.shape[0], 1), 10)))

# Calculate the distance between users and the BS
dist_BS = np.linalg.norm(user_coords - bs_coord, axis=1)

dist_UAV_BS = np.linalg.norm(drone_coords - bs_coord, axis=1)

# Calculate the distance between users and drones
dist_drones = np.linalg.norm(user_coords[:, np.newaxis, :] - drone_coords, axis=2)
dist_drones1 = np.linalg.norm(user_coords[:, np.newaxis, :] - drone_coords1, axis=2)

Np = len(user_positions_type1_xy)
N = len(user_positions_type1_xy) + len(user_positions_type2_xy)

# example coordinates for users, drones and the base station
bs_coord = np.array([bs1,bs2,25])
user_coords = np.hstack((user_positions_type1_xy, np.full((len(user_positions_type1_xy), 1), 1.5)))
user_coords1 = np.hstack((user_positions_type2_xy, np.full((len(user_positions_type2_xy), 1), 1.5)))
drone_coords = np.hstack((uav_positions, np.full((len(uav_positions), 1), 10)))
drone_coords1 = np.hstack((uav_positions1, np.full((len(uav_positions1), 1), 10)))

# calculate the distance between users and the BS
dist_BS = np.linalg.norm(user_coords - bs_coord, axis=1)
dist_BS1 = np.linalg.norm(user_coords1 - bs_coord, axis=1)
dist_UAV_BS = np.linalg.norm(drone_coords - bs_coord, axis=1)
dis_UAV1_BS = np.linalg.norm(drone_coords1 - bs_coord, axis=1)

# calculate the distance between users and drones
dist_drones = np.linalg.norm(user_coords[:, np.newaxis, :] - drone_coords, axis=2)
dis_drones_ext= np.linalg.norm(user_coords[:, np.newaxis, :] - drone_coords1, axis=2)
dist_drones1 = np.linalg.norm(user_coords1[:, np.newaxis, :] - drone_coords, axis=2)
dist_drones1_ext = np.linalg.norm(user_coords1[:, np.newaxis, :] - drone_coords1, axis=2)




# Maximum supported SNR in dB
max_snr_dB = 19.6
# Convert max_snr_dB to the linear scale
max_snr = 10 ** (max_snr_dB / 10)



# Negative binomial distribution parameters
r = 10  # Number of successes (assumption, adjust as needed)
u = 0.01  # Probability of success (assumption, adjust as needed)

# Truncated negative binomial parameters
a, b = 1, 600

total_slots = int(360/0.001)
slot_duration = 0.001
observation_time = 1 * 60 * 6
B_total = 100e6

average_data_rate1 = np.zeros(Np)
average_data_rate2 = np.zeros(N - Np)



def truncated_negbinom(n, p, a, b):
    '''Generates a sample from a truncated negative binomial distribution.'''
    x = np.arange(a, b+1)
    probs = scipy_stats.nbinom.pmf(x, n, p)
    probs /= probs.sum()
    return scipy_stats.rv_discrete(values=(range(a, b+1), probs)).rvs()


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

# Main simulation loop
for t in range(1, total_slots + 1):
    # Generate the number of active users using the negative binomial distribution
    X1 = truncated_negbinom(r, u, a, b)
    X2 = truncated_negbinom(r, u, a, b)

    # Randomly choose active user indices
    active_type1_indices = np.random.choice(Np, X1, replace=False)
    active_type2_indices = np.random.choice(N - Np, X2, replace=False)

    drone_association = np.zeros(len(active_type1_indices), dtype=bool)
    for idx, user_index in enumerate(active_type1_indices):
        user_drone_snr = SNR_drones_ext[user_index]
        user_bs_snr = SNR_drones[user_index]
        drone_bs_snr = 0

        min_uav_snr = np.min(user_drone_snr + drone_bs_snr)
    

    
        drone_association[idx] = user_bs_snr <= min_uav_snr

    B_total = 100e6  # Bandwidth is 100 MHz
    resources_per_user = B_total / (X1 + X2)

    # Calculate PF metric for each active user
    pf_metrics = np.zeros(X1 + X2)
    for i, active_user_index in enumerate(np.concatenate((active_type1_indices, active_type2_indices + Np))):
        if active_user_index < Np:
            user_index = active_user_index
            shape = 1
            scale = 1000000
            bits = np.random.gamma(shape, scale)
            if drone_association[np.where(active_type1_indices == user_index)[0][0]]:
                snr_UAV_UE = SNR_drones_ext[user_index]
                achievable_data_rate_array = resources_per_user * np.log2(1 + snr_UAV_UE)
                max_data_rate = np.max(achievable_data_rate_array)
                r_ui = (max_data_rate + remaining_bits1[user_index]) / np.maximum(1e-10, R_avg1[user_index])
            else:
                snr = SNR_drones[user_index]
                snr = np.minimum(snr, max_snr)
                achievable_data_rate = resources_per_user * np.log2(1 + snr)
                r_ui = (achievable_data_rate + remaining_bits1[user_index]) / np.maximum(1e-10, R_avg1[user_index])
        else:
            user_index = active_user_index - Np
            mean = 4
            sigma = 0.7
            bits = np.random.lognormal(mean, sigma)
            snr = SNR_drones1[user_index]
            snr = np.minimum(snr, max_snr)
            achievable_data_rate = resources_per_user * np.log2(1 + snr)
            r_ui = (achievable_data_rate + remaining_bits2[user_index]) / np.maximum(1e-10, R_avg2[user_index])
        pf_metrics[i] = r_ui

    # Sort active user indices based on PF metric
    active_user_indices_sorted = np.concatenate((active_type1_indices, active_type2_indices + Np))[np.argsort(pf_metrics)[::-1]]

    # Iterate over sorted active users
    for active_user_index in active_user_indices_sorted:
        if active_user_index < Np:
            user_index = active_user_index
            shape = 1
            scale = 1000000
            bits = np.random.gamma(shape, scale)
            if drone_association[np.where(active_type1_indices == user_index)[0][0]]:
                snr_UAV_UE = SNR_drones_ext[user_index]
                achievable_data_rate_array = resources_per_user * np.log2(1 + snr_UAV_UE)
                
                highest_drone_idx = np.argmax(achievable_data_rate_array)
                achievable_data_rate = achievable_data_rate_array[highest_drone_idx]
                R_avg1[user_index] = (1 - 1 / (t + 1e-10)) * R_avg1[user_index] + (1 / (t + 1e-10)) * achievable_data_rate
                transmitted_bits = np.minimum(bits, achievable_data_rate * slot_duration)
                D1[user_index] += transmitted_bits
                remaining_bits1[user_index] = bits - transmitted_bits
            else:
                snr = SNR_drones[user_index]
                snr = np.minimum(snr, max_snr)
                achievable_data_rate = resources_per_user * np.log2(1 + snr)
                R_avg1[user_index] = (1 - 1 / (t + 1e-10)) * R_avg1[user_index] + (1 / (t + 1e-10)) * achievable_data_rate
                transmitted_bits = np.minimum(bits, achievable_data_rate * slot_duration)
                D1[user_index] += transmitted_bits
                remaining_bits1[user_index] = bits - transmitted_bits
            met_demand1[user_index] += transmitted_bits
            demand1[user_index] += bits
        else:
            user_index = active_user_index - Np
            mean = 4
            sigma = 0.7
            bits = np.random.lognormal(mean, sigma)
            snr = SNR_drones1[user_index]
            snr = np.minimum(snr, max_snr)
            achievable_data_rate = resources_per_user * np.log2(1 + snr)
            R_avg2[user_index] = (1 - 1 / (t + 1e-10)) * R_avg2[user_index] + (1 / (t + 1e-10)) * achievable_data_rate
            transmitted_bits = np.minimum(bits, achievable_data_rate * slot_duration)
            D2[user_index] += transmitted_bits
            remaining_bits2[user_index] = bits - transmitted_bits
            met_demand2[user_index] += transmitted_bits
            demand2[user_index] += bits

# Compute total demands and the amount met
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
user_ids_type1_s12 = np.arange(1, Np + 1)
user_ids_type2_s12 = np.arange(Np + 1, N + 1)

# Set up the plot
fig, ax = plt.subplots(figsize=(15, 6))
bar_width = 0.4
opacity = 0.8

    # Create bar plot for Type 1 users
type1_bars = ax.bar(user_ids_type1_s12 - bar_width / 2, individual_data_rate_type1_s12_mbps, bar_width,
                    alpha=opacity, color='blue', label="Live Stream Users")

# Create bar plot for Type 2 users
type2_bars = ax.bar(user_ids_type2_s12 + bar_width / 2, individual_data_rate_type2_s12_mbps, bar_width,
                    alpha=opacity, color='red', label="Non-Live Stream Users")

# Create bar plot for Type 1 users
#type1_bars = ax.bar(user_ids_type1_s12 - bar_width / 2, individual_data_rate_type1_s12_mbps, bar_width, alpha=opacity, label="Live Stream Users")

# Create bar plot for Type 2 users
#type2_bars = ax.bar(user_ids_type2_s12 + bar_width / 2, individual_data_rate_type2_s12_mbps, bar_width, alpha=opacity, label="Non-Live Stream 2 Users")

# Customize the plot
ax.set_xlabel("User IDs")
ax.set_ylabel("Average Data Rate (Mbps)")
ax.legend()
plt.show()


print("Average Data Rate for Type 1 Users: {:.8f} Mbps".format(data_rate_type1_s12_mbps))
print("Average Data Rate for Type 2 Users: {:.8f} Mbps".format(data_rate_type2_s12_mbps))
print("Network Throughput: {:.8f} Mbps".format(network_throughput_s12_mbps))
print("Met Demand for Type 1 Users: {:.8f}%".format(met_demand_percentage_type1))
print("Met Demand for Type 2 Users: {:.8f}%".format(met_demand_percentage_type2))

# Graphical representation of results
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.plot(D1, label='Demand')
plt.plot(met_demand1, label='Met demand')
plt.title('Type 1 users')
plt.xlabel('User index')
plt.ylabel('Bits')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(D2, label='Demand')
plt.plot(met_demand2, label='Met demand')
plt.title('Type 2 users')
plt.xlabel('User index')
plt.ylabel('Bits')
plt.legend()

plt.tight_layout()
plt.show()


