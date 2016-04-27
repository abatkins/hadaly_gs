# hadaly_gs

### Hadaly Gridsearch 

## Run MPI

## vortex
<pre>
qsub -l walltime=30:00:00 -l nodes=1:c18b:ppn=12 runscript_opt
qsub -I -l walltime=10:00 -l nodes=1:c18x:ppn=6


mpirun -np 6 python-mpi ./hadaly_grid.py --nested
</pre>

## hurricane / whirlwind
<pre>
qsub -l walltime=30:00:00 -l nodes=2:c11x:ppn=6 runscript_xeon
qsub -I -l walltime=20:00 -l nodes=2:c11x:ppn=6
qsub -I -l walltime=5:00 -l nodes=1:x5672:ppn=4
mvp2run python-mpi ./hadaly_grid.py -f aiddata_full.csv --nested
</pre>
