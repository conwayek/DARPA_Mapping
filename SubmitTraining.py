import os
import glob


#image_dir = '/scratch/e.conway/DARPA_MAPS/Validation/'
#clue_dir = '/scratch/e.conway/DARPA_MAPS/clue_CSVs/'
#result_dir = '/scratch/e.conway/DARPA_MAPS/ValidationResults/'
#files = glob.glob(image_dir+'*.tif')

image_dir = '/scratch/e.conway/DARPA_MAPS/Training/'
clue_dir = '/scratch/e.conway/DARPA_MAPS/CluesTesting/'
result_dir = '/scratch/e.conway/DARPA_MAPS/Results/'
files = glob.glob(clue_dir+'*.csv')


#print(len(files))
nfiles = len(files)
for i in range(nfiles):
    file = files[i]
    #jobname = os.path.basename(file).split('.tif')[0]+'.sh'
    #geo_name = int(os.path.basename(file).split('.tif')[0].split('GEO_')[1])
    jobname = os.path.basename(file).split('.csv')[0]+'.sh'
    geo_name = int(os.path.basename(file).split('_clue.csv')[0].split('GEO_')[1])
    outname = os.path.join(result_dir,os.path.basename(file).split('_clue.csv')[0]+'.csv')
    #"""
    with open(jobname,'w') as f:
        f.write('#!/bin/bash'+'\n')
        f.write('#SBATCH --nodes=1'+'\n')
        f.write('#SBATCH --time=1:0:00'+'\n')
        f.write('#SBATCH --job-name=i'+str(geo_name)+'\n')
        #f.write('#SBATCH --partition=short'+'\n')
        f.write('#SBATCH --partition=gpu'+'\n')
        f.write('#SBATCH --gres=gpu:1'+'\n')
        f.write('#SBATCH --mem=16GB'+'\n')
        f.write('#SBATCH --output=slurm-'+str(geo_name)+'.out'+'\n')
        f.write('module load anaconda3/2022.01'+'\n')
        f.write('source activate'+'\n')
        f.write('source activate darpa2'+'\n')
        f.write('conda activate darpa2'+'\n')
        f.write('python3 DARPA_Color.py '+str(result_dir)+' '+str(image_dir)+' '+str(os.path.basename(file).split('_clue.csv')[0]+'.tif')\
                +' '+str(clue_dir)+'\n')
    if(os.path.exists(outname)==False):
        os.system("sbatch "+str(jobname))
    #"""
    
    #if(os.path.exists(outname)==False):
    #    print(outname)
    #    os.system('python3 DARPA_Color.py '+str(result_dir)+' '+str(image_dir)+\
    #              ' '+str(os.path.basename(file).split('_clue.csv')[0]+'.tif')\
    #            +' '+str(clue_dir))
        
