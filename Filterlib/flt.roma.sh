clear

LD=$HOME/local/intel/compilers_and_libraries_2018.1.163/linux/ipp/lib/intel64:$HOME/local/gcc-5.4/lib64:$LD_LIBRARY_PATH
NU=4096
NV=4096
PROJS=32
SRC_DIR=$HOME/digital-phantom/
DST_DIR=$HOME/dump/filtered\_w$NU\_h$NV\_c$PROJS/
THREADS=1
EXE=./Release/Filterlib

cmd="LD_LIBRARY_PATH=$LD $EXE $NU $NV $PROJS $SRC_DIR $DST_DIR $THREADS"
echo $cmd && LD_LIBRARY_PATH=$LD $EXE $NU $NV $PROJS $SRC_DIR $DST_DIR $THREADS

