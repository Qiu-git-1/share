import os
import numpy as np
#/cs/labs/peleg/nivc/Palach/iclr2024/sinbad_runs/sbatches/
out_path = "./sinbad_runs/sbatches/"
'''
Import the required Python libraries.
Set the output path out_path to store the generated shell script.
A series of parameters are defined, such as dataset_names, number of projections n_projections_list, number of quantiles n_quantiles_list, pyramid_level_list, Clip size ratio patch_size_ratio_list, shrink factor shrinkage_factor_list, superblock size super_patch_n_list, and number of repetitions rep_list.
Iterating through all combinations of parameters through a nested loop generates a list named name_list, containing all possible combinations of parameters.
For each element in the name_list, a separate shell script is generated that contains the commands needed to run the SINBAD algorithm. These commands include setting environment variables, activating the virtual environment, switching the working directory to where the SINBAD code is located, and calling the Python script sinbad_single_layer.py with the appropriate parameters.
Generate a master batch script, batch_master.sh, that uses the SLURM job scheduler to run all generated shell scripts in parallel. The --array parameter specifies the range of the script array, the --exclude parameter is used to exclude certain nodes, the --gres and --mem parameters specify GPU resources and memory resources respectively, the -c and -A parameters specify the number of CPU cores and SLURM accounts respectively, and the --time parameter sets the maximum running time of the job.
Generate a bishop.sh script that is invoked at the start of the batch job and executes the shell script corresponding to the ID.
'''

dataset_names = ['breakfast_box_loco','juice_bottle_loco','pushpins_loco','screw_bag_loco','splicing_connectors_loco',
                 'breakfast_box_struct','juice_bottle_struct','pushpins_struct','screw_bag_struct','splicing_connectors_struct']


n_projections_list = [1000]
n_quantiles_list = [5]


pyramid_level_list = [7,14]
patch_size_ratio_list = [0.25,0.5,0.7,0.99]
shrinkage_factor_list = [0.1]
super_patch_n_list = [4]
rep_list = [0]

name_list = []

for dataset_name in dataset_names:
    for n_projections in n_projections_list:
        for n_quantiles in n_quantiles_list:
                for pyramid_level in pyramid_level_list:
                    for patch_size_ratio in patch_size_ratio_list:
                            for shrinkage_factor in shrinkage_factor_list:
                                    for super_patch_n in super_patch_n_list:
                                        for rep in rep_list:
                                            name_list.append([dataset_name, n_projections, n_quantiles, pyramid_level, patch_size_ratio, shrinkage_factor,super_patch_n, rep])

for i in range(len(name_list)):

    f  = open(out_path + str(i) + '.sh', 'w')
    f.write("""#! /bin/bash
export LD_LIBRARY_PATH=/usr/lib/llvm-7/lib/
cd /cs/labs/peleg/nivc/Palach/iclr2024/SINBAD
python sinbad_single_layer.py --version_name "ver1"  --mvtype %s --n_projections %d --n_quantiles %d   --net wide_resnet50_2   --pyramid_level %d  --crop_size_ratio %f --shrinkage_factor %f   --crop_num_edge  %d  --rep %d """
            %( name_list[i][0], name_list[i][1], name_list[i][2], name_list[i][3], name_list[i][4], name_list[i][5], name_list[i][6], name_list[i][7]))
    f.close()


#f  = open(out_path + "batch_master" + '.sh', 'w')
#f.write("sbatch --array=0-%d%s10 --exclude=cyril-01,ampere-01,binky-01,binky-02,binky-03,binky-04,binky-05,binky-06,arion-01,arion-02,drape-01,drape-02 --gres=gpu:1,vmem:10g --mem=18000m -c2 --time=1-12 -A yedid /cs/labs/peleg/nivc/Palach/iclr2024/sinbad_runs/sbatches/bishop.sh"%(len(name_list),'%'))

with open(out_path + "batch_master" + '.sh', "w") as f:
    f.write("#!/bin/bash\n")
    f.write("for i in {1..79}\n")
    f.write("do\n")
    f.write("    /root/autodl-tmp/sinbad_runs/sbatches/${i}.sh\n")#Remember to change the address
    f.write("done\n")




f.close()
