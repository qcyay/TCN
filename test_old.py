import argparse
from typing import List
import torch
from utils.config_utils import load_config
from models.tcn import TCN
from dataset_loaders.dataloader import TcnDataset

# parse CLI arguments
parser = argparse.ArgumentParser()
parser.add_argument("--config_path", type = str, default = "configs.TCN.default_config", help = "File path to config file for loading and testing pretrained TCN model.")
parser.add_argument("--device", type = str, default = "cpu", help = "Device to host model and data.")
args = parser.parse_args()

# load config
config = load_config(args.config_path)


def load_model(device: torch.device):
	'''Creates TCN and loads pretrained weights.'''
	model_info = torch.load(config.model_path, map_location = device)
	state_dict = model_info["state_dict"]
	del model_info["state_dict"]
	tcn = TCN(**model_info).to(device)
	tcn.load_state_dict(state_dict)
	return tcn


def print_results(trial_names: List[str],
					label_names: List[str],
					estimates: torch.FloatTensor,
					labels: torch.FloatTensor,
					model_history: int,
					trial_sequence_lengths: List[float]):
	'''Prints model RMSE relative to ground-truth.'''
	for i, trial_name in enumerate(trial_names):
		print(f"{trial_name} results:")
		for j, label_name in enumerate(label_names):
			# Extract estimates and labels. Ignore andy starting or ending sequences that used zero padding.
			estimate = estimates[i, j, model_history:trial_sequence_lengths[i]]
			label = labels[i, j, model_history:trial_sequence_lengths[i]]

			# Correct for any intentional delays in model estimates
			if config.model_delays[j] != 0:
				estimate = estimate[config.model_delays[j]:]
				label = label[:-config.model_delays[j]]

			# Ignore data points corresponding to nans in input or label data
			valid_index = torch.where(~torch.isnan(estimate) & ~torch.isnan(label))
			estimate = estimate[valid_index]
			label = label[valid_index]

			# Compute and print RMSE
			rmse = torch.sqrt(torch.mean((estimate - label)**2))
			print(f"{label_name} RMSE: {rmse} Nm/kg")


def main():
	# Load TCN
	device = torch.device(args.device)
	tcn = load_model(device)

	# Disable gradient tracking and training parameters
	tcn.train(False)

	# Load data after replacing * with desired side from config
	input_names = [name.replace("*", config.side) for name in config.input_names]
	label_names = [name.replace("*", config.side) for name in config.label_names]
	dataset = TcnDataset(data_dir = config.data_dir,
							input_names = input_names,
							label_names = label_names,
							side = config.side,
							participant_masses = config.participant_masses,
							device = device,
							mode = 'test')
	# input_data，尺寸为[batch_size, num_input_features, max_sequence_length]
	# label_data，尺寸为[batch_size, num_label_features, max_sequence_length]
	# trial_sequence_lengths，列表，每个trial数据的原始长度
	input_data, label_data, trial_sequence_lengths = dataset[:]

	# Compute model estimates
	with torch.no_grad():

		# Forward pass
		out = tcn(input_data)

		# Compute RMSE per trial and print
		print_results(dataset.get_trial_names(), label_names, out, label_data, tcn.get_effective_history(), trial_sequence_lengths)


if __name__=="__main__":
	main()
