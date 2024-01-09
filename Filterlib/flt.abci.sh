clear

IPP_HOME=$INTEL_HOME/ipp

echo IPP_HOME=$IPP_HOME
LD=$IPP_HOME/lib/intel64 #:$HOME/local/gcc-5.4/lib64:$LD_LIBRARY_PATH

NU=2048
NV=2048
PROJS=240
SRC_DIR=/groups2/gaa50004/gaa10008ku/data/
DST_DIR=/groups2/gaa50004/gaa10008ku/dump/filtered\_w$NU\_h$NV\_c$PROJS/
THREADS=80
EXE=./Release/Filterlib

cmd="LD_LIBRARY_PATH=$LD $EXE $NU $NV $PROJS $SRC_DIR $DST_DIR $THREADS"
echo $cmd && LD_LIBRARY_PATH=$LD $EXE $NU $NV $PROJS $SRC_DIR $DST_DIR $THREADS


