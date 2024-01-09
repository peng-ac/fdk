#!/bin/sh

export TMI_CONFIG=~/tmi.conf

TARGET_NAME=cuFDK
pkill -f $TARGET_NAME
#./repeat.sh

mode=Release

#LD_PATH=$HOME/local/cuda-9.0/lib64:$LOCAL_HOME/openmpi-3.1.0/lib64:$(echo $LD_LIBRARY_PATH)
#MPIRUN=$(which mpirun)
MPIRUN=~/local/intel/compilers_and_libraries_2018.1.163/linux/mpi/intel64/bin/mpiexec
# RUN=$MPIRUN -x LD_LIBRARY_PATH=$LD_PATH --hostfile hostfile.txt -np 27 -npernode 9 ./$mode/mnfdk

NP=2
NPP=2
ROWS=1
VOL_SIZE=512
VOL_NX=$VOL_SIZE
VOL_NY=$VOL_SIZE
VOL_NZ=$VOL_SIZE
IMG_SIZE=512
NU=$IMG_SIZE
NV=$IMG_SIZE
PROJS=2048
SRC_DIR=$HOME/data/
DST_DIR=$HOME/dump/
MAX_PROJ_DB=16384

echo mode=$mode
echo MPIRUN=$MPIRUN
echo NPP=$NPP
echo MAX_PROJ_DB=$MAX_PROJ_DB

#$MPIRUN --hostfile hostfile.txt -np 4 ./Release/mnfdk
#$MPIRUN -x LD_LIBRARY_PATH=$LD_PATH --hostfile hostfile.txt -np 36 ./Debug/mnfdk

# export I_MPI_DAPL_UD=enable

export LD_LIBRARY_PATH=$HOME/local/cuda-9.0/lib64:$HOME/local/intel/compilers_and_libraries_2018.1.163/linux/mpi/lib64:$LD_LIBRARY_PATH

$MPIRUN --hostfile hostfile.txt -np $NP -ppn $NPP ./$mode/$TARGET_NAME $NPP $ROWS $VOL_NX $VOL_NY $VOL_NZ $NU $NV $PROJS $SRC_DIR $DST_DIR $MAX_PROJ_DB
#$MPIRUN -x LD_LIBRARY_PATH=$LD_PATH -np 9 ./Debug/mnfdk
