LD=$HOME/local/intel/compilers_and_libraries_2018.1.163/linux/ipp/lib/intel64:$HOME/local/gcc-5.4/lib64:$LD_LIBRARY_PATH
LD_LIBRARY_PATH=$LD ./Release/Filterlib ../../data/phantom3d/shepp-logan_w2048_h2048_c1200/ ../../data/phantom3d/filtered-shepp-logan_w2048_h2048_c1200/ 1
#./Release/Filterlib ../../data/proj-img/ ../../data/dump/filtered-proj-img/ 32
