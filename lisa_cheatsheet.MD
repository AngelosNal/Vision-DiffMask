- Copy the whole folder:
```scp -r /path/to/here/ lcur____@lisa.surfsara.nl:/path/to/there/```

- Load the modules to have conda:
```
module load 2021
module load Anaconda3/2021.05
```

- Open a gpu node with a terminal: (1 hour to install stuff, blocking call to cluster thus once you get assigned a node you won't get kicked out)
```srun -p gpu_shared_course -n 1 --mem=32000M --ntasks-per-node 1 --gpus 1 --cpus-per-task 2 -t 1:00:00 --pty /bin/bash```

> User should change to lcur____@"random_string"

- Install the environment:
```
cd Patch-DiffMask
conda env create -f environment.yml
```

- Exit the node and reenter to restart the shell (conda shenanigans)
```
exit
srun -p gpu_shared_course -n 1 --mem=32000M --ntasks-per-node 1 --gpus 1 --cpus-per-task 2 -t 1:00:00 --pty /bin/bash
conda activate dl2
```


- Activate the environment:
```conda activate dl2```

- Run the code:
```python main.py```

- If you wanna set up a train job:
TODO
