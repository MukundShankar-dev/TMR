import os
from datetime import datetime
import argparse
import time
import pandas as pd
import socket


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
parser.add_argument('--nhrs', type=int, default=5)
parser.add_argument('--base-dir', default=f'{os.getcwd()}')
parser.add_argument('--output-dirname', default='output')
parser.add_argument('--dryrun', action='store_true')
parser.add_argument('--rerun', action='store_true')
parser.add_argument('--explore', action='store_true')


parser.add_argument('--qos', default="scav", type=str, help='Qos to run')
parser.add_argument('--env', default="dtw_scores", type=str, help = "Set the name of the dir you want to dump")
parser.add_argument('--gpu', default=1, type=int, help='Number of gpus')
parser.add_argument('--cores', default=4, type=int, help='Number of cpu cores')
parser.add_argument('--mem', default=5, type=int, help='RAM in G')
parser.add_argument('--gpu_type', default='none', type=str, help='RAM in G')


# parser.add_argument('--path', default='/fs/vulcan-projects/actionbytes/vis/ab_training_run3_rerun_32_0.0001_4334_new_dl_nocasl_checkpoint_best_dmap_ab_info.hkl')
# parser.add_argument('--num_ab', default= 100000, type=int, help='number of actionbytes')

args = parser.parse_args()

feat_dump = True

args = parser.parse_args()
args.env += str(int(time.time()))


output_dir = os.path.join(args.base_dir, args.output_dirname, args.env)
if not os.path.exists(output_dir):
    os.makedirs(output_dir)



print("Output Directory: %s" % output_dir)
per_job = 100
total_ids = 27448

params = [(start_idx) for  start_idx in range(0, total_ids, per_job)]

print(len(params))
pca = True
                        
temporal_skip = None
hostname = socket.gethostname()
with open(f'{args.base_dir}/output/{args.env}/now.txt', "w") as nowfile,\
     open(f'{args.base_dir}/output/{args.env}/log.txt', "w") as output_namefile,\
     open(f'{args.base_dir}/output/{args.env}/err.txt', "w") as error_namefile,\
     open(f'{args.base_dir}/output/{args.env}/name.txt', "w") as namefile:

    parameters = []

    for i, (start_idx) in enumerate(params):
        if os.path.isfile(f'dtw_scores/all_dtw_{start_idx}_{start_idx+100}.csv') == False:
            parameters.append(start_idx)
            parameters.append(start_idx+50)
    
    for i, (start_idx) in enumerate(parameters):
        now = datetime.now()
        datetimestr = now.strftime("%m%d_%H%M:%S.%f")
        name = f'test_{i}'
        cmd = f'python calculate_scores_matrix.py --start_idx {start_idx}'
        cmd += f' --per_job_ids 50 --num_cores {args.cores}'        
        
        nowfile.write(f'{cmd}\n')
        namefile.write(f'{(os.path.join(output_dir, name))}.log\n')
        output_namefile.write(f'{(os.path.join(output_dir, name))}_log.txt\n')
        error_namefile.write(f'{(os.path.join(output_dir, name))}_error.txt\n')

        #break
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
    #slurmfile.write(f"#SBATCH --array=1-10\n")
    slurmfile.write("#SBATCH --output=/dev/null\n")
    slurmfile.write("#SBATCH --error=/dev/null\n")
    slurmfile.write("#SBATCH --requeue\n")
    # slurmfile.write("#SBATCH --exclude=vulcan[00-23]\n")
    
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
            # if hostname in {'nexus', 'vulcan'}:
            if args.gpu_type == 'a4':
                gpu_str = 'rtxa4000:'
            elif args.gpu_type == 'a6':
                gpu_str = 'rtxa6000:'
            elif args.gpu_type == 'a5':
                gpu_str = 'rtxa5000:'
            else:
                gpu_str = ''
            slurmfile.write(f'#SBATCH --gres=gpu:{gpu_str}{args.gpu}\n')
           
    else:
       
        slurmfile.write("#SBATCH --account=abhinav\n")
        slurmfile.write("#SBATCH --qos=%s\n" % args.qos)
        slurmfile.write("#SBATCH --time=%d:00:00\n" % args.nhrs)
        slurmfile.write("#SBATCH --gres=gpu:%d\n" % args.gpu)
        slurmfile.write("#SBATCH --cpus-per-task=%d\n" % args.cores)
        slurmfile.write("#SBATCH --mem=%dG\n" % args.mem)
    # if 'nexus' in hostname:
        # slurmfile.write("#SBATCH --exclude=legacygpu[00-08]\n")
    # elif 'vulcan' in hostname:
        # print('vulcan')
        # slurmfile.write("#SBATCH --exclude=brigid[00-03],brigid[06-12],brigid14\n")


    
    slurmfile.write("\n")
    #slurmfile.write("export MKL_SERVICE_FORCE_INTEL=1\n")p
    slurmfile.write("cd " + os.getcwd() + '\n')
    slurmfile.write("export MKL_THREADING_LAYER=GNU\n")
    slurmfile.write("source /vulcanscratch/mukunds/anaconda3/bin/activate\n")
    slurmfile.write("conda activate tmr\n")
    # slurmfile.write("cd ./libs/utils\n")
    # slurmfile.write("python setup.py install --user\n")
    # slurmfile.write("cd ../..\n")


    slurmfile.write(f"srun --output=$(head -n $SLURM_ARRAY_TASK_ID {args.base_dir}/output/{args.env}/log.txt | tail -n 1) --error=$(head -n $SLURM_ARRAY_TASK_ID {args.base_dir}/output/{args.env}/err.txt | tail -n 1)  $(head -n $SLURM_ARRAY_TASK_ID {args.base_dir}/output/{args.env}/now.txt | tail -n 1)\n")
    slurmfile.write("\n")
print(slurm_command)
print("Running on {}, with {} gpus, {} cores, {} mem for {} hour".format(args.qos, args.gpu, args.cores, args.mem , args.nhrs))
if not args.dryrun:
   os.system("%s &" % slurm_command)