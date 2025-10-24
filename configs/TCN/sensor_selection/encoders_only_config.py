import os

# relative file path to trained model
model_path = os.path.join("logs", "trained_tcn_encoders_only.tar")

# relative path to data
# data_dir = os.path.join("..", "data")
data_dir = 'data/example'

# corresponding leg (model is not dependent on side)
side = "r"

# corresponding model input names in dataset (* is substituted with side)
input_names = ["hip_angle_*", "hip_angle_*_velocity_filt", 
				"knee_angle_*", "knee_angle_*_velocity_filt"]

# corresponding model label names in dataset
label_names = ["hip_flexion_*_moment", "knee_angle_*_moment"]

# intentional model delay (in data points)
model_delays = [10, 0] # hip moment estimates are delayed by 50 ms

# participant masses for normalizing insole forces.
# - NOTE: This is a simplification. Detailed participant masses are provided in the readme of the corresponding dataset.
participant_masses = {
	"BT01": 80.59,
	"BT02": 72.24,
	"BT03": 95.29,
	"BT04": 98.23,
	"BT06": 79.33,
	"BT07": 64.49,
	"BT08": 69.13,
	"BT09": 82.31,
	"BT10": 93.45,
	"BT11": 50.39,
	"BT12": 78.15,
	"BT13": 89.85,
	"BT14": 67.30,
	"BT15": 58.40,
	"BT16": 64.33,
	"BT17": 60.03,
	"BT18": 67.96,
	"BT19": 69.95,
	"BT20": 55.44,
	"BT21": 58.85,
	"BT22": 76.79,
	"BT23": 67.23,
	"BT24": 77.79
}

