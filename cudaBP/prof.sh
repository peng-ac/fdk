clear \
CUDA_HOME=~/local/cuda-9.0 LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH PATH=$CUDA_HOME/bin:$PATH nvprof --print-gpu-trace ./Release/cudaBP 512 512 32 1024 1024 2048 16384 ~/data/ ~/dump/ 0