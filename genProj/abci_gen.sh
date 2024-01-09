DST=/fs2/groups2/gaa50008/gaa10008ku/data/
SRC=$HOME/data/phantom3d/Shepp-Logan-512x512x512.vol
LD=$LOCAL_HOME/cuda-9.0/lib64:$LD_LIBRARY_PATH
LD_LIBRARY_PATH=$LD ./Release/genProj $DST_DIR $SRC_VOL 1024 1024 32
echo ""--------------------------------------------------
mkdir -p $DST
LD=$LOCAL_HOME/cuda-9.0/lib64:$LOCAL_HOME/gcc-5.4/lib64:$LD_LIBRARY_PATH
#!/bin/bash
for w in 512 1024 2048 4096 8192
do
	for n in 512 1024 2048 4096 8192 16384
	do
		LD_LIBRARY_PATH=$LD ./Release/genProj $DST $SRC $w $w $n
	done
done