# hadaly_gs

### Hadaly Gridsearch 

## Run MPI

## vortex
<pre>
qsub -I -l walltime=10:00 -l nodes=1:c18x:ppn=3
mpirun -np 3 python-mpi ./hadaly_grid.py --prod --nested
</pre>
 
## hurricane / whirlwind
<pre>
qsub -I -l walltime=30:00 -l nodes=2:x5672:ppn=8
mvp2run python-mpi ./hadaly_grid.py --prod
</pre>
