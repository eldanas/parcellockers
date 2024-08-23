import os
import subprocess
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
from itertools import product

# Run parameters
sp_stds = [0.25, 0.125]  # sp_std parameter
delay_stds = [0.25, 0.125]  # delay parameter
assigment_policies = ["lower_bound", "bl1", "bl3", "preference"]  # assignment policy parameter
forecast_methods = ["pessimistic", "expected"]  # forecast method parameter
pickup_distribution = [0.3, 0.25, 0.2, 0.15, 0.1]
arrival_rates = [0.04, 0.048]  # arrival rate parameter
days = 50
delivery_time = 0.75
sp_data_file = "data/linz-sp.csv"
dp_data_file = "data/linz-dp.csv"
output_dir = "output"
concurrent_workers = 4  # number of concurrent workers
max_attempts = 1  # number of attempts to attempt running the command (in case of failure)

pickup_distribution_str = " ".join([str(x) for x in pickup_distribution])
run_params = []
command_prefix = "python sim.py"
constant_args = f"--days {days} --sp_data_file {sp_data_file} " \
               f"--dp_data_file {dp_data_file} --delivery_time {delivery_time} " \
               f"--pickup_distribution {pickup_distribution_str}" \
               f" --output_dir {output_dir}"

for arrival_rate in arrival_rates:
    for sp_std, delay_std in product(sp_stds, delay_stds):
        for assignment_method in assigment_policies:
            param = [arrival_rate, sp_std, delay_std, assignment_method]
            if assignment_method == "preference":
                for forecast_method in forecast_methods:
                    run_params.append(param + [forecast_method])
            else:
                run_params.append(param)

if not os.path.exists("output/logs"):
    os.makedirs("output/logs")


def run_command(params):
    params_str = f"{constant_args} " \
                 f"--arrival_rate {params[0]} " \
                 f"--sp_std {params[1]} --delay_std {params[2]} " \
                 f"--assignment_policy {params[3]}"
    if len(params) == 5:
        params_str = f"{params_str} --forecast_method {params[4]}"
    command = f"{command_prefix} {params_str}"
    log_path = f"output/logs/{arrival_rate}_{"_".join(map(str, params))}.log"
    attempts = 0
    with open(log_path, "w") as f:
        success = False
        while not success and attempts < max_attempts:
            print(f"Running command with params: {params}")
            process = subprocess.run(command, shell=True, stdout=f, stderr=f)
            success = process.returncode == 0
            attempts += 1
            if success:
                print(f"\033[92m")  # green
                print(f"Command with params {params} succeeded.")
            else:
                print(f"\033[91m")  # red
                print(f"Command with params {params} failed.")
            print(f"\033[0m")


with ProcessPoolExecutor(max_workers=concurrent_workers, mp_context=mp.get_context("fork")) as executor:
    executor.map(run_command, run_params)

