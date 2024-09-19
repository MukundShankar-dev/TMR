import os
from datetime import datetime
import argparse
import time
import pandas as pd
import socket
from itertools import product


qos_dict = {"sailon" : {"nhrs" : 2, "cores": 16, "mem":128},
            "scav" : {"nhrs" : 72, "cores": 16, "mem":128},
            "vulc_scav" : {"nhrs" : 24, "cores": 16, "mem":128},
            "cml_scav" : {"nhrs" : 24, "cores": 16, "mem":128}, 

            "high" : {"gpu":4, "cores": 16, "mem":120, "nhrs": 36},
            "medium" : {"gpu":2, "cores": 8, "mem":64, "nhrs": 72},
            "default" : {"gpu":1, "cores": 4, "mem":32, "nhrs": 168},
            "tron" : {"gpu":1, "cores": 4, "mem":32, "nhrs": 168}}

def check_qos(args):
    for key, max_value in qos_dict[args.qos].items():
        val_from_args = getattr(args, key)
        if val_from_args != None:
            if val_from_args > max_value:
                raise ValueError("Invalid paramter for {} for {}".format(key, args.qos))
        else:
            setattr(args, key, max_value)
    return args

#TODO: Add day funtionality too 
parser = argparse.ArgumentParser()
parser.add_argument('--nhrs', type=int, default=48)
parser.add_argument('--base-dir', default=f'{os.getcwd()}')
parser.add_argument('--output-dirname', default='output')
parser.add_argument('--dryrun', action='store_true')
parser.add_argument('--rerun', action='store_true')
parser.add_argument('--explore', action='store_true')


parser.add_argument('--qos', default="scav", type=str, help='Qos to run')
parser.add_argument('--env', default="flag_dtw", type=str, help = "Set the name of the dir you want to dump")
parser.add_argument('--gpu', default=1, type=int, help='Number of gpus')
parser.add_argument('--cores', default=16, type=int, help='Number of cpu cores')
parser.add_argument('--mem', default=32, type=int, help='RAM in G')
parser.add_argument('--gpu_type', default='none', type=str, help='RAM in G')

args = parser.parse_args()

feat_dump = True

args = parser.parse_args()
args.env += str(int(time.time()))

output_dir = os.path.join(args.base_dir, args.output_dirname, args.env)
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

print("Output Directory: %s" % output_dir)

# NOTE: USE FOR RUNNING LMD HYPERPARAMETER SWEEP
# all_lmd_contrastive = [0.1, 0.25, 0.5, 0.75, 0.9]
# all_lmd_dtw = [0.1, 0.25, 0.5, 0.75, 0.9]
# params = list(product(all_lmd_contrastive, all_lmd_dtw))

# NOTE: USE FOR RUNNING SWEEP ON DTW LMD WEIGHT SWEEP
# params = [10.0, 5.0, 25.0, 40.0, 50.0, 75.0, 100.0]

# NOTE: USE FOR RUNNING MARGIN HYPERMARAMETER SWEEP
# params = [0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.5]

# params = [(2, 5), (5, 8), (10, 13)]

# params = [50, 100, 200, 300, 400, 500]

params = list(range(2000, 7200, 100))

pca = True              
temporal_skip = None
hostname = socket.gethostname()
with open(f'{args.base_dir}/output/{args.env}/now.txt', "w") as nowfile,\
     open(f'{args.base_dir}/output/{args.env}/log.txt', "w") as output_namefile,\
     open(f'{args.base_dir}/output/{args.env}/err.txt', "w") as error_namefile,\
     open(f'{args.base_dir}/output/{args.env}/name.txt', "w") as namefile:
    
    for start_idx in params:
        now = datetime.now()
        datetimestr = now.strftime("%m%d_%H%M:%S.%f")

        name = f"start_idx_{start_idx}"

        cmd = f'python calculate_scores_matrix_flag.py --num_cores {args.cores} --start_idx {start_idx} --per_job_ids 100'
        nowfile.write(f'{cmd}\n')
        namefile.write(f'{(os.path.join(output_dir, name))}.log\n')
        output_namefile.write(f'{(os.path.join(output_dir, name))}_log.txt\n')
        error_namefile.write(f'{(os.path.join(output_dir, name))}_error.txt\n')

    # NOTE: if using both losses, to jointly tune both
    # for lmd_contrastive, lmd_dtw in params:
    #     now = datetime.now()
    #     datetimestr = now.strftime("%m%d_%H%M:%S.%f")
        
    #     name = f'test_cont_{lmd_contrastive}_dtw_{lmd_dtw}'

    #     cmd = f'python train.py run_dir=outputs/tmr_cont_{lmd_contrastive}_dtw_{lmd_dtw} '
    #     cmd += f'model.run_dir=outputs/tmr_cont_{lmd_contrastive}_dtw_{lmd_dtw} '
    #     cmd += f'model.lmd.dtw={lmd_dtw} model.lmd.contrastive={lmd_contrastive} '
    #     cmd += 'model.dtw_loss_type=\"cosine\" model.use_dtw=True '
    #     cmd += f'model.dtw_margin=0.15 '
    #     cmd += f'model.wandb_name=\"cont_{lmd_contrastive},dtw_{lmd_dtw}\"'
        
    #     nowfile.write(f'{cmd}\n')
    #     namefile.write(f'{(os.path.join(output_dir, name))}.log\n')
    #     output_namefile.write(f'{(os.path.join(output_dir, name))}_log.txt\n')
    #     error_namefile.write(f'{(os.path.join(output_dir, name))}_error.txt\n')

    # NOTE: to tune lmd DTW
    # for lmd_weight in params:
    #     now = datetime.now()
    #     datetimestr = now.strftime("%m%d_%H%M:%S.%f")
        
    #     name = f'dtw_lmd_{lmd_weight}'

    #     cmd = f'python train.py run_dir=outputs/{name} '
    #     cmd += f'model.run_dir=outputs/{name} '
    #     cmd += f'model.lmd.dtw={lmd_weight} model.lmd.contrastive=0.0 '
    #     cmd += 'model.dtw_loss_type=\"cosine\" model.use_dtw=True '
    #     cmd += f'model.dtw_margin=0.15 model.use_contrastive=False '

    #     cmd += f'lower=5 upper=8 '
    #     cmd += f'model.wandb_name=\"{name}\"'
        
    #     nowfile.write(f'{cmd}\n')
    #     namefile.write(f'{(os.path.join(output_dir, name))}.log\n')
    #     output_namefile.write(f'{(os.path.join(output_dir, name))}_log.txt\n')
    #     error_namefile.write(f'{(os.path.join(output_dir, name))}_error.txt\n')

    # NOTE: to tune triplet loss margin
    # for margin in params:
    #     now = datetime.now()
    #     datetimestr = now.strftime("%m%d_%H%M:%S.%f")
        
    #     name = f'tmr_cos_loss_{margin}'
    #     cmd = f'python train.py run_dir=outputs/tmr_cos_loss_{margin} '
    #     cmd += f'model.run_dir=outputs/tmr_cos_loss_{margin} '
    #     cmd += f'model.lmd.dtw=1.0 model.lmd.contrastive=0.1 '
    #     cmd += 'model.dtw_loss_type=\"cosine\" model.use_dtw=True '
    #     cmd += f'model.dtw_margin={margin} '
    #     cmd += f'model.wandb_name=\"TMR_cos_0.15_correct\"'
        
    #     nowfile.write(f'{cmd}\n')
    #     namefile.write(f'{(os.path.join(output_dir, name))}.log\n')
    #     output_namefile.write(f'{(os.path.join(output_dir, name))}_log.txt\n')
    #     error_namefile.write(f'{(os.path.join(output_dir, name))}_error.txt\n')

    # NOTE: to tune negative lower and upper
    # for lower, upper in params:
    #     now = datetime.now()
    #     datetimestr = now.strftime("%m%d_%H%M:%S.%f")

    #     name = f'neg_{lower}_{upper}'
    #     cmd = f'python train.py run_dir=outputs/{name} '
    #     cmd += f'model.run_dir=outputs/{name} '
    #     cmd += f'model.lmd.dtw=5.0 model.lmd.contrastive=0 model.dtw_loss_type=\"cosine\" '
    #     cmd += f'model.use_dtw=True model.dtw_margin=0.15 model.wandb_name={name} model.use_contrastive=False '
    #     cmd += f'lower={lower} upper={upper}'
    #     nowfile.write(f'{cmd}\n')
    #     namefile.write(f'{(os.path.join(output_dir, name))}.log\n')
    #     output_namefile.write(f'{(os.path.join(output_dir, name))}_log.txt\n')
    #     error_namefile.write(f'{(os.path.join(output_dir, name))}_error.txt\n')

    # NOTE: to play with DTW thresholds
    # for dtw_thresh in params:
    #     now = datetime.now()
    #     datetimestr = now.strftime("%m%d_%H%M:%S.%f")

    #     name = f'dtw_threshold_{dtw_thresh}'
    #     cmd = f'python train.py run_dir=outputs/dtw_threshold_{dtw_thresh} '
    #     cmd += f'model.run_dir=outputs/dtw_threshold_{dtw_thresh} '
    #     cmd += f'model.lmd.dtw=0.9 model.lmd.contrastive=0.9 model.dtw_loss_type=\"cosine\" '
    #     cmd += f'model.use_dtw=True model.dtw_margin=0.15 model.wandb_name={name} model.use_contrastive=False '
    #     cmd += f'lower=10 upper=13 threshold_dtw={dtw_thresh}'
    #     nowfile.write(f'{cmd}\n')
    #     namefile.write(f'{(os.path.join(output_dir, name))}.log\n')
    #     output_namefile.write(f'{(os.path.join(output_dir, name))}_log.txt\n')
    #     error_namefile.write(f'{(os.path.join(output_dir, name))}_error.txt\n')

###########################################################################
# Make a {name}.slurm file in the {output_dir} which defines this job.
#slurm_script_path = os.path.join(output_dir, '%s.slurm' % name)

start=1
slurm_script_path = os.path.join(output_dir, f'dtw.slurm')
slurm_command = "sbatch %s" % slurm_script_path

# Make the .slurm file
with open(slurm_script_path, 'w') as slurmfile:
    slurmfile.write("#!/bin/bash\n")
    slurmfile.write(f"#SBATCH --array=1-{len(params)}\n")
    # slurmfile.write(f"#SBATCH --array=1-{1}\n")
    slurmfile.write("#SBATCH --output=/dev/null\n")
    slurmfile.write("#SBATCH --error=/dev/null\n")
    slurmfile.write("#SBATCH --requeue\n")
    
    args = check_qos(args)

    if "scav" in args.qos or "tron" in args.qos:
        if args.qos == "scav":
            slurmfile.write("#SBATCH --account=scavenger\n")
            slurmfile.write("#SBATCH --qos scavenger\n")
            slurmfile.write("#SBATCH --partition scavenger\n")

        elif args.qos == "vulc_scav":
            slurmfile.write("#SBATCH --account=vulcan-abhinav\n")
            slurmfile.write("#SBATCH --qos vulcan-scavenger\n")
            slurmfile.write("#SBATCH --partition vulcan-scavenger\n")
        
        elif args.qos == "cml_scav":
            slurmfile.write("#SBATCH --account=cml-scavenger\n")
            slurmfile.write("#SBATCH --qos cml-scavenger\n")
            slurmfile.write("#SBATCH --partition cml-scavenger\n")
 
        slurmfile.write("#SBATCH --time=%d:00:00\n" % args.nhrs)
        slurmfile.write("#SBATCH --cpus-per-task=%d\n" % args.cores)
        slurmfile.write("#SBATCH --mem=%dG\n" % args.mem)

        if not args.gpu is None:
            if args.gpu_type == 'a4':
                gpu_str = 'rtxa4000:'
            elif args.gpu_type == 'a6':
                gpu_str = 'rtxa6000:'
            elif args.gpu_type == 'a5':
                gpu_str = 'rtxa5000:'
            else:
                gpu_str = 'rtx2080ti:'
            slurmfile.write(f'#SBATCH --gres=gpu:{gpu_str}{args.gpu}\n')
           
    else:
       
        slurmfile.write("#SBATCH --account=abhinav\n")
        slurmfile.write("#SBATCH --qos=%s\n" % args.qos)
        slurmfile.write("#SBATCH --time=%d:00:00\n" % args.nhrs)
        slurmfile.write("#SBATCH --gres=gpu:%d\n" % args.gpu)
        slurmfile.write("#SBATCH --cpus-per-task=%d\n" % args.cores)
        slurmfile.write("#SBATCH --mem=%dG\n" % args.mem)

    slurmfile.write("\n")
    slurmfile.write("cd " + os.getcwd() + '\n')
    slurmfile.write("export MKL_THREADING_LAYER=GNU\n")
    slurmfile.write("source /vulcanscratch/mukunds/anaconda3/bin/activate\n")
    slurmfile.write("conda activate tmr\n")
    slurmfile.write("export HYDRA_FULL_ERROR=1\n")

    slurmfile.write(f"srun --output=$(head -n $SLURM_ARRAY_TASK_ID {args.base_dir}/output/{args.env}/log.txt | tail -n 1) --error=$(head -n $SLURM_ARRAY_TASK_ID {args.base_dir}/output/{args.env}/err.txt | tail -n 1)  $(head -n $SLURM_ARRAY_TASK_ID {args.base_dir}/output/{args.env}/now.txt | tail -n 1)\n")
    slurmfile.write("\n")
print(slurm_command)
print("Running on {}, with {} gpus, {} cores, {} mem for {} hour".format(args.qos, args.gpu, args.cores, args.mem , args.nhrs))
if not args.dryrun:
   os.system("%s &" % slurm_command)