__Under construction__

# Documentation to run keras training on CSCS Piz Daint
[CSCS Piz Daint](http://user.cscs.ch/computing_systems/piz_daint/index.html)
is a great resource to perform deep learning training and we provide here a
set of instructions to run [Keras](https://keras.io/) with the
[Theano](https://github.com/Theano/Theano) backend (the Tensorflow backend is
not supported yet).
We provide setup to run [distributed training over
mpi](https://github.com/duanders/mpi_learn) of keras models.
We provide setup to run [hyperparameter optimization using
Spearmint](https://github.com/HIPS/Spearmint) over slurm.
We provide further tools to combine all of the above.

# Keras setup
## Theano backend
The python libraries are installed once under a virtual environment with the piz daint gpu setting
```
module load daint-gpu

module load Python/3.5.2-CrayGNU-2016.11
module load pycuda/2016.1.2-CrayGNU-2016.11-Python-3.5.2-cuda-8.0
module load h5py/2.6.0-CrayGNU-2016.11-Python-3.5.2-serial

virtualenv kerasP3
source kerasP3/bin/activate

pip install tables
pip install keras --upgrade
pip install scikit-learn
pip install mpi4py
```

The environement can be used anytime by activating the virtual env
```
source kerasP3/bin/activate
```

## Tensorflow backend 
This is not supported yet

# Distributed Training
We provide here instructions to run the [mpi_learn
package](https://github.com/duanders/mpi_learn).

```
git clone git@github.com:vlimant/mpi_learn.git
cd mpi_learn 
git branch python_3
```

# Hyperparameter Optimization

# Running on Piz Daint
## Keras Runtest
One can run the default [mnist example]() provided by Francois Chollet.
First get the script
```
wget https://raw.githubusercontent.com/fchollet/keras/master/examples/mnist_cnn.py
```
Then run it over slurm. Create a running script `mnist_submit`
```
#!/bin/bash -l

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --ntasks-per-core=1
#SBATCH --cpus-per-task=1
#SBATCH --constraint=gpu
#SBATCH --time=00:30:00

export CRAY_CUDA_MPS=1

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
source kerasP3/bin/activate

srun python ~/mnist_cnn.py
```

and submit it to the batch system using
```
sbatch mnist_submit
```

## MPI Learn Runtest

## Spearmint Runtest
