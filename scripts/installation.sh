conda create -n  my_custom_python_39 python=3.9 -y
source ~/anaconda3/bin/activate
conda activate my_custom_python_39
conda install mpi4py -y
pip install -e .