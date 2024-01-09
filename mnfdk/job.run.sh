#!/bin/sh

#$-l rt_F=34
#$-cwd
#$-l h_rt=6:00:00

source /etc/profile.d/modules.sh

export LOCAL_HOME=$HOME/local
export LD_LIBRARY_PATH=$LOCAL_HOME/lib
export IPP_HOME=$LOCAL_HOME/intel/compilers_and_libraries_2018.1.163/linux/ipp
export PATH=$LOCAL_HOME/sshpass-1.05:$LOCAL_HOME/htop-2.2.0:$LOCAL_HOME/cmake/bin:$LOCAL_HOME/bin:$PATH
export PATH=$LOCAL_HOME/gcc-5.4/bin:$PATH
export LD_LIBRARY_PATH=$LOCAL_HOME/gcc-5.4/lib64:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$LOCAL_HOME/intel/compilers_and_libraries_2018.1.163/linux/ipp/lib/intel64:$LD_LIBRARY_PATH

export MPIROOT=$LOCAL_HOME/openmpi-3.1.0
export PATH=$MPIROOT/bin:$PATH
export LD_LIBRARY_PATH=$MPIROOT/lib:$LD_LIBRARY_PATH
RUN=$MPIRUN -np 27 -npernode 9 ./$mode/mnfdk
export MANPATH=$MANPATH:$MPIROOT/share/man

echo PATH=$PATH

LD_PATH=$HOME/local/cuda-9.0/lib64:$LOCAL_HOME/openmpi-3.1.1/lib64:$(echo $LD_LIBRARY_PATH)
export LD_LIBRARY_PATH=$LD_PATH

echo LD_LIBRARY_PATH=$LD_LIBRARY_PATH
echo LOCAL_PATH=$LOCAL_HOME
echo HOME=$HOME

MPIRUN=$(which mpirun)

mode=Release
echo MPIRUN=$MPIRUN
$MPIRUN -np 90 -npernode 9 ./$mode/mnfdk
# $MPIRUN -x LD_LIBRARY_PATH=$LD_PATH -np 27 -npernode 9 ./$mode/mnfdk


