 
NU=512
NV=512
PROJS=128
SRC_DIR=$HOME/digital-phantom/
DST_DIR=$HOME/dump/filtered\_w$NU\_h$NV\_c$PROJS/
BATCH_COUNT=1
EXE=./Release/FilterlibCufft


CUDA_HOME=~/local/cuda-9.0 PATH=$CUDA_HOME/bin:$PATH LD_LIBRARY_PATH=$CUDA_HOME/lib64/:$LD_LIBRARY_PATH
 $EXE $NU $NV $PROJS $SRC_DIR $DST_DIR $BATCH_COUNT
