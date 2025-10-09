import os
import json

this_file_path = os.path.dirname(os.path.realpath(__file__))
# current_path = os.getcwd()

def submit_job(gen_name, mkt="RT", test_run=True):

    # create a directory to save job scripts
    job_scripts_dir = os.path.join(this_file_path, "job_scripts")
    if not os.path.isdir(job_scripts_dir):
        os.mkdir(job_scripts_dir)
    if test_run:
        title = "Test"
    else:
        title = "Sweep"

    file_name = os.path.join(job_scripts_dir, f"{title}_run_PT_opt_{gen_name}_{mkt}"  + ".sh")
    # conda_env_path = os.path.join(current_path, "..", "..", ".conda", "envs", "idaes")
    conda_env_path = "idaes"
    with open(file_name, "w") as f:
        f.write(
            "#!/bin/bash\n"
            + "#$ -M xchen24@nd.edu\n"
            + "#$ -m ae\n"
            + "#$ -q long\n"
            + "#$ -N " + f"{title}_run_PT_opt_{gen_name}_{mkt}" + "\n"
            + f"conda activate {conda_env_path}\n"
            + "export LD_LIBRARY_PATH=~/.conda/envs/regen/lib:$LD_LIBRARY_PATH \n"
            + "module load gurobi\n"
            + "module load ipopt/3.14.2 \n"
            + f"python ./run_PT_opt.py {gen_name} {mkt}\n"
        )

    os.system(f"qsub {file_name}")


if __name__ == "__main__":
    
    # gen_path = os.path.join("..", "Data", "gen_dict.json")
    # with open(gen_path, 'rb') as f:
    #     gen_dict = json.load(f)
    # gen_names = list(gen_dict["fossil"].keys())
    # for n in gen_names:
    #     submit_job(n, test_run=False)

    submit_job("101_STEAM_3", mkt="DA", test_run=True)