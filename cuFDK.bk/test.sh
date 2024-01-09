#!/bin/sh

TARGET_NAME=cuFDK
pkill -f $TARGET_NAME
./repeat.sh

mode=Release

LD_PATH=$HOME/local/cuda-9.0/lib64:$LOCAL_HOME/openmpi-3.1.0/lib64:$(echo $LD_LIBRARY_PATH)
MPIRUN=$(which mpirun)
# RUN=$MPIRUN -x LD_LIBRARY_PATH=$LD_PATH --hostfile hostfile.txt -np 27 -npernode 9 ./$mode/mnfdk

echo mode=$mode
echo MPIRUN=$MPIRUN


#$MPIRUN --hostfile hostfile.txt -np 4 ./Release/mnfdk
#$MPIRUN -x LD_LIBRARY_PATH=$LD_PATH --hostfile hostfile.txt -np 36 ./Debug/mnfdk

$MPIRUN -x LD_LIBRARY_PATH=$LD_PATH --hostfile hostfile.txt -np 6 -npernode 2 ./$mode/$TARGET_NAME
#$MPIRUN -x LD_LIBRARY_PATH=$LD_PATH -np 9 ./Debug/mnfdk