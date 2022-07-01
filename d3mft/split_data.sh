#!/bin/bash

if [ -d data_tmp ];then    
    rm -rf data_tmp    
    mkdir data_tmp
else
    mkdir data_tmp
fi 

if [ -d data_original ]; then
    rm -rf data
    mv data_original data
fi

# read the following from the config.ini
beta=`grep 'beta' config.ini | cut -d= -f2- | cut -c 2-`
#basis=`grep 'basis' config.ini | cut -d= -f2- | cut -c 2-`
basis="tau"
num_per_file=`grep 'n_samples_per_file' config.ini | cut -d= -f2- | cut -c 2-`

# split params
split -l ${num_per_file} -d --additional-suffix=.csv data/params_${beta}_00.csv data_tmp/params_${beta}_

# split delta_tau
split -l ${num_per_file} -d --additional-suffix=_${basis}.csv data/Delta_${beta}_00_${basis}.csv data_tmp/Delta_${beta}_
# split G_PT_tau
split -l ${num_per_file} -d --additional-suffix=_${basis}.csv data/G_PT_${beta}_00_${basis}.csv data_tmp/G_PT_${beta}_
# split G_SC_tau
split -l ${num_per_file} -d --additional-suffix=_${basis}.csv data/G_SC_${beta}_00_${basis}.csv data_tmp/G_SC_${beta}_
# split G_ED_Q_tau
split -l ${num_per_file} -d --additional-suffix=_${basis}.csv data/G_ED_Q_${beta}_00_${basis}.csv data_tmp/G_ED_Q_${beta}_

basis="legendre"

# split delta_legendre
split -l ${num_per_file} -d --additional-suffix=_${basis}.csv data/Delta_${beta}_00_${basis}.csv data_tmp/Delta_${beta}_
# split G_PT_legendre
split -l ${num_per_file} -d --additional-suffix=_${basis}.csv data/G_PT_${beta}_00_${basis}.csv data_tmp/G_PT_${beta}_
# split G_SC_legendre
split -l ${num_per_file} -d --additional-suffix=_${basis}.csv data/G_SC_${beta}_00_${basis}.csv data_tmp/G_SC_${beta}_
# split G_ED_Q_legendre
split -l ${num_per_file} -d --additional-suffix=_${basis}.csv data/G_ED_Q_${beta}_00_${basis}.csv data_tmp/G_ED_Q_${beta}_

cp -r data data_original
rm -rf data
mv data_tmp data
