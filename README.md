# hadaly_gs

### Hadaly Gridsearch 

## Install mpi4py
Anaconda:
<pre>
conda install mpi4py
<pre>

Pip:
<pre>
pip install mpi4py
</pre>

Note: You will also have to install the open-mpi package for your os. 

## Sciclone Modules
In the home directory you must edit your cshrc files in the home directory. 

Make sure has .cshrc.rhel6-xeon only the following lines:
<pre>
module load isa/nehalem
module load python/2.7.5/gcc
module load acml/5.1.0/gcc
module load mpi4py/1.3.1/gcc
module add scipy/0.12.0
</pre>

Make sure has .cshrc.rhel6-opteron only the following lines:
<pre>
module load isa/seoul
module load acml/5.3.1/gcc
module load mpi4py/1.3.1/gcc
</pre>

## Run MPI

## vortex
<pre>
qsub -l walltime=30:00:00 -l nodes=1:c18b:ppn=12 runscript_opt
qsub -I -l walltime=10:00 -l nodes=1:c18x:ppn=6

mpirun -np 6 python-mpi ./hadaly_grid.py --gridsearch nested -j your_job
</pre>

## hurricane / whirlwind
<pre>
qsub -l walltime=30:00:00 -l nodes=2:c11x:ppn=6 runscript_xeon
qsub -I -l walltime=20:00 -l nodes=2:c11x:ppn=6
qsub -I -l walltime=5:00 -l nodes=1:x5672:ppn=4
mvp2run python-mpi ./hadaly_grid.py -f aiddata_full.csv --gridsearch nested -j your_job
</pre>
