[Controlling Repair Crews dispatch towards EV chargers resilience using multi-agent reinforcement learning]

This directory is composed of the main files I have developed to complete my Master thesis. 
Usage

•	Before running the training, make sure to adjust the “arguments.py” file, by assigning the appropriate values to the parameters defined in the script. In particular, Make sure to change the absolute path attached to the “model_dir” and “results_dir”, where the trained models and the model’s performance measurements are stored respectively. 

•	In order to variate the type of RL algorithm underlying the trained model, change the RL algorithm name attributed to the “alg” parameter in the “arguments.py” file. The name assigned to this parameter must match one of the file names specified in the “algorithms” folder

