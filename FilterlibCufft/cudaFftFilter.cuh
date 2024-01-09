#ifndef __cudaFftFilter_H
#define __cudaFftFilter_H

#include "cudaBackProjection.cuh"


bool FilterRow_Real32(const DataRef<float>& src, DataRef<float>& dst, const DataRef<float>& filter);

bool CorrectImage_Real32(const DataRef<float>& src, const DataRef<float>& table, DataRef<float>& dst);




#endif //__cudaFftFilter_H