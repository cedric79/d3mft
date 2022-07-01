#!/bin/bash
conda create -n $1 -y python=3.7 > install.log
eval "$(conda shell.bash hook)"
conda activate $1
echo "In ${env_name}"
which python
python --version
echo "--------- Installing TRIQS ---------"
conda install -y -c conda-forge triqs >> install.log
echo "--------- Installing Numba ---------"
conda install -y numba=0.48.0 >> install.log
echo "--------- Installing Quspin ---------"
conda install -y -c weinbe58 quspin >> install.log
echo "--------- Installing Pandas ---------"
conda install -y pandas >> install.log
echo "--------- Installing tables ---------"
pip install tables >> install.log 
echo "--------- Installing Scikit-learn ---------"
pip install scikit-learn >> install.log
echo "--------- Installing Tensorflow ---------"
pip3 install tensorflow==2.3.1 >> install.log
echo "--------- Installing Jupyter ---------"
conda install -y -c conda-forge notebook >> install.log
conda install -y nb_conda >> install.log
echo "--------- Installing LAIM ---------"
pip install . >> install.log 
