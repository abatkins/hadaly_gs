# hadaly_gs

### Hadaly Gridsearch 

## Run MPI

## vortex
<pre>
qsub -I -l walltime=20:00 -l nodes=1:c18x:ppn=4
mpirun -np 4 python-mpi ./hadaly_grid.py --prod --nested
</pre>
 
## hurricane / whirlwind
<pre>
qsub -I -l walltime=5:00 -l nodes=1:x5672:ppn=4
mvp2run python-mpi ./hadaly_grid.py --prod --nested
</pre>
