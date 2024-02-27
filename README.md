# iFDK: A Scalable Framework for Instant High-resolution Image Reconstruction (SC' 19)
We use ABCI (https://abci.ai/) HPC to solve the high-resolution image reconstruction problem, e.g. 2048^3, 4096^3, 8192^3.
This repository contains the artifact, e.g. job scripts, benchmarks.

## DEPENDENCIES
We tested on ABCI, using Nvidia Volta V100 GPUs, with GCC 4.8 and NVCC 9.0.
The following libraries and tools are requirements:

    cmake = 3.1
    CUDA = 9.0
    python >= 2.7
    Intel MPI 2018.2.199
    Intel IPP 2018.2.199
    Malab R2018a
    Insight Segmentation and Registration Toolkit (ITK)
    Reconstruction Toolkit (RTK)

## BUILD/RUN-TIME ENVIRONMENT


## SOURCE CODE
- Our Source Code is Closed.

- RTK-1.4.0 BP kernel with single precision
the improved kernel can be found in 

	src/RTK/rtkCudaFDKBackProjectionImageFilter.cu

## DATA

- Generate 3D shepp-logan phantom by Matlab script as

        tools/phantom3d/phantom3d.m
	
	a sample (size 512^3, single precison, raw data) can be downloaded from dropbox as
	
	https://www.dropbox.com/s/o0xgt4igipdve2l/Shepp-Logan-512x512x512.vol?dl=0

- Generate 2D projection

        ./generate-projections.sh    


## How to run

- all modules are in folder bin

- Generating job script by run scrip in folder jobs/generate-jobs as

        python gen-jobs.py strong2k
        python gen-jobs.py strong4k
        python gen-jobs.py strong8k
        python gen-jobs.py weak2k
        python gen-jobs.py weak4k
        python gen-jobs.py weak8k

- Run jobs in the Root-folder of iFDK-archifact
	
		./run.sh all
		./run.sh strong2k
		./run.sh strong4k
		./run.sh strong8k
		./run.sh weak2k
		./run.sh weak4k
		./run.sh weak8k

- Run benchmarks




## BENCHMARKS
The related benchmarks can be found in follows:

https://www.openrtk.org/

https://github.com/LLNL/ior

https://docs.nvidia.com/cuda/cuda-samples/index.html

## tools

https://jp.mathworks.com/matlabcentral/fileexchange/9416-3d-shepp-logan-phantom?s_tid=mwa_osa_a

https://www.openrtk.org/    



## Help/Support:
For more information or questions, contact the authors at chinhou0718#gmail.com (replace # by @, please)
