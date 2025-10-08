import os
import json
from fossil_rolling_horizon_PT_parameter import period, scenario, horizon

this_file_path = os.path.dirname(os.path.realpath(__file__))

def submit_job(
    gen_name,
    period,
    scenario,
    horizon,
):

    # create a directory to save job scripts
    job_scripts_dir = os.path.join(this_file_path, "sim_job_scripts")
    if not os.path.isdir(job_scripts_dir):
        os.mkdir(job_scripts_dir)

    file_name = os.path.join(job_scripts_dir, f"Rolling_horizon_Pricetaker_{gen_name}_p_{period}_s_{scenario}_h_{horizon}.sh")
    with open(file_name, "w") as f:
        f.write(
            "#!/bin/bash\n"
            + "#$ -M xchen24@nd.edu\n"
            + "#$ -m ae\n"
            + "#$ -q long\n"
            + f"#$ -N Rolling_horizon_Pricetaker_{gen_name}_p_{period}_s_{scenario}_h_{horizon}\n"
            + "conda activate idaes\n"
            # + "export LD_LIBRARY_PATH=~/.conda/envs/regen/lib:$LD_LIBRARY_PATH \n"
            + "module load gurobi\n"
            # + "module load ipopt/3.14.2 \n"
            + f"python run_rolling_horizon_PT.py {gen_name}\n"
        )

    os.system(f"qsub {file_name}")


if __name__ == "__main__":

    # with open(os.path.join(this_file_path, "..", "Data", "gen_dict.json"), "r") as f:
    #     gen_dict = json.load(f)
    
    # fossil_gen_names = list(gen_dict["fossil"].keys())
    # for gen_name in fossil_gen_names:
    #     submit_job(gen_name, period, scenario, horizon)

    # Example to submit a job
    gen_name = "101_STEAM_3"
    submit_job(gen_name, period, scenario, horizon)
