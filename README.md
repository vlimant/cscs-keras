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

The installation of theano is pulled in when installing Keras. There is an
experimental build for cscs.

## Tensorflow backend 

There is a tensorflow build for CSCS but there is a destructive interference
with Keras when running on the cluster.

# Distributed Training
The nstructions get the [mpi_learn
package](https://github.com/duanders/mpi_learn) is available on github, and we
pull it in in the python environement setup below.

# Hyperparameter Optimization

## Mongodb 

Install mongodb from a compiled version
```
mkdir mongodb
cd mongodb
wget https://fastdl.mongodb.org/linux/mongodb-linux-x86_64-suse12-3.4.2.tgz
tar zxvf mongodb-linux-x86_64-suse12-3.4.2.tgz

```

and run a mongodb instance from the scratch directory

```
mkdir $SCRATCH/mongodb
 mongodb/mongodb-linux-x86_64-suse12-3.4.2/bin/mongod --fork --dbpath
 $SCRATCH/mongodb/ --logpath
 $SCRATCH/mongodb/db.log
```
## Spearmint

The instructions are available [from spearmint on
github](https://github.com/HIPS/Spearmint) and is pulled in in the python
environement setup below.


# Python envirronment

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
pip install pymongo
cd kerasP3
git clone https://github.com/HIPS/Spearmint.git
pip install -e $PWD/Spearmint
cd -
```

The environement can be used anytime by activating the virtual env
```
source kerasP3/bin/activate
```
or with python2

```
module load daint-gpu
module load h5py/2.6.0-CrayGNU-2016.11-Python-2.7.12-serial
module load pycuda/2016.1.2-CrayGNU-2016.11-Python-2.7.12-cuda-8.0.54

virtualenv kerasP2
source kerasP2/bin/activate

pip install tables
pip install keras --upgrade
pip install scikit-learn
pip install mpi4py
pip install pymongo

cd kerasP2
git clone https://github.com/HIPS/Spearmint.git
pip install -e $PWD/Spearmint
cd -
```

and use it thereafter

```
source kerasP2/bin/activate
```


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

Alpha instructions [are provided
there](https://github.com/vlimant/mpi-learn-benchmark) and are still experimental.

## Spearmint Runtest

This is massively failing on python3 because the package is not python2
compatible.
Effort is on-going to make a python3 port.
