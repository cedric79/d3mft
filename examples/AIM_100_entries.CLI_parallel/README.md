To run
mpirun -np 10 python setup_database.py
mpirun -np 10 python solve_database.py
mpirun -np 1 python train_database.py

You can compare results with
../AIM_100_entries.CLI_parallel.ref
which should be qualitatively similar, but due to the different random distributions they will be differ quantitatively