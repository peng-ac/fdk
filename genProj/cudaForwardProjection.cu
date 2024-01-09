#include "cudaIterType.cuh"

texture<float, cudaTextureType3D, cudaReadModeElementType> tex_vol_roi(false, cudaFilterModePoint, cudaAddressModeClamp);
texture<float, cudaTextureType3D, cudaReadModeElementType> tex_vol_mask(false, cudaFilterModePoint, cudaAddressModeClamp);

__constant__ float2 c_rot_det_xy_tbl[1024];
__constant__ float c_rot_det_z_tbl[1024];

struct RotDetTbl{
	__device__ __forceinline__
	float3 operator()(int x, int y) const{
		assert(x>=0 && x < sizeof(c_rot_det_xy_tbl)/sizeof(c_rot_det_xy_tbl[0]));
		assert(y>=0 && y < sizeof(c_rot_det_z_tbl)/sizeof(c_rot_det_z_tbl[0]));
		return make_float3(c_rot_det_xy_tbl[x].x, c_rot_det_xy_tbl[x].y, c_rot_det_z_tbl[y]);
	}
};

struct cuRay
{
    float3 o;   // origin
    float3 d;   // direction
};


// intersect ray with a box
// http://www.siggraph.org/education/materials/HyperGraph/raytrace/rtinter3.htm

__device__ __host__ __forceinline__
int cuIntersectBox(cuRay r, float3 boxmin, float3 boxmax, float *tnear, float *tfar)
{
    // compute intersection of ray with all six bbox planes
    float3 invR = make_float3(1.0f) / r.d;
    float3 tbot = invR * (boxmin - r.o);
    float3 ttop = invR * (boxmax - r.o);

    // re-order intersections to find smallest and largest on each axis
    float3 tmin = fminf(ttop, tbot);
    float3 tmax = fmaxf(ttop, tbot);

    // find the largest tmin and the smallest tmax
    float largest_tmin = fmaxf(fmaxf(tmin.x, tmin.y), fmaxf(tmin.x, tmin.z));
    float smallest_tmax = fminf(fminf(tmax.x, tmax.y), fminf(tmax.x, tmax.z));

    *tnear = largest_tmin;
    *tfar = smallest_tmax;

    return smallest_tmax > largest_tmin;
}

static void TestIntersectBox(){
	cuRay r;
	r.o = make_float3(0, 0, 4);
	r.d = make_float3(0, 0, -1);
	float3 boxmin = make_float3(-1, -1, -1);
	float3 boxmax = make_float3(1, 1, 1);
	float tnear = 0;
	float tfar = 0;

	cuIntersectBox(r, boxmin, boxmax, &tnear, &tfar);
}

__global__ void kernel_forwardProject(float2   fCosSin,
									  float3   src_pos,
									  cuParameter para,	
									  cuVolumeGeo vol_geo_roi,
									  cuVolumeGeo vol_geo_mask,
									  float*   dev_proj,
									  int      pitch_dev_proj											 								 						
									  )
{

  unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
  unsigned int j = blockIdx.y*blockDim.y + threadIdx.y;

  if (i >= para.nu || j >= para.nv)
    return;

  float det_yMM = para.SID - para.SOD;
  float src_yMM = -para.SOD; 
  //float tStep = MIN3(para.dx, para.dy, para.dz);    //mm unit
  float fsum;
  float3 rot_pt1_mm;
  // Setting ray origin
  cuRay ray;
  ray.o = src_pos;

  cuDetectorCoor<float> detectorCor(para.du, para.dv, para.nu, para.nv, para.offset_nu, para.offset_nv);
  cuVolumeCoor<float>  volCor_roi(vol_geo_roi.dx, vol_geo_roi.dy, vol_geo_roi.dz, vol_geo_roi.nx, vol_geo_roi.ny, vol_geo_roi.nz);		  
  cuVolumeCoor<float>   volCor_mask(vol_geo_mask.dx, vol_geo_mask.dy, vol_geo_mask.dz, vol_geo_mask.nx, vol_geo_mask.ny, vol_geo_mask.nz);
  {
	  //detector
	  float u, v, uMM, vMM;
	  detectorCor.ij2xy(i, j, u, v);
	  detectorCor.pixel2mm(u, v, uMM, vMM);
	  float3 pt1_mm = make_float3(uMM, det_yMM, vMM);
	  cuRotate(float(0), float(0), fCosSin.x, fCosSin.y, pt1_mm.x, pt1_mm.y, rot_pt1_mm.x, rot_pt1_mm.y); 
	  rot_pt1_mm.z = pt1_mm.z;
  }
  ray.d = cuNormize(rot_pt1_mm - src_pos); 

  {
	  float sum_mask = 0;
	  float sum_roi  = 0;
	  float tnear_roi, tfar_roi, tnear_mask, tfar_mask, t;
	  float tstep_roi = vol_geo_roi.tStep;
	  float tstep_mask = vol_geo_mask.tStep;
	  bool bHit_roi, bHit_mask;
	  bHit_mask = cuIntersectBox(ray, vol_geo_mask.boxmin, vol_geo_mask.boxmax, &tnear_mask, &tfar_mask);
	  if (bHit_mask){
		  bHit_roi  = cuIntersectBox(ray, vol_geo_roi.boxmin, vol_geo_roi.boxmax, &tnear_roi, &tfar_roi);
		  float3 ray_ori = ray.o;		  
		  float3 pos;
		  float3 xyz, cur_point_ijk;
		  float val;

		  float3 vStep_roi, vStep_mask;
		  float3 src_ijk_roi, src_ijk_mask;
		  {
			  volCor_roi.mm2pixel(0, 0, 0, xyz.x, xyz.y, xyz.z);
			  volCor_roi.xyz2ijk(xyz.x, xyz.y, xyz.z, src_ijk_roi.x, src_ijk_roi.y, src_ijk_roi.z);
			  volCor_roi.mm2pixel(ray.d.x, ray.d.y, ray.d.z, xyz.x, xyz.y, xyz.z);
			  volCor_roi.xyz2ijk(xyz.x, xyz.y, xyz.z, cur_point_ijk.x, cur_point_ijk.y, cur_point_ijk.z);
			  vStep_roi = cur_point_ijk - src_ijk_roi;

			  volCor_roi.mm2pixel(ray_ori.x, ray_ori.y, ray_ori.z, xyz.x, xyz.y, xyz.z);
			  volCor_roi.xyz2ijk(xyz.x, xyz.y, xyz.z, src_ijk_roi.x, src_ijk_roi.y, src_ijk_roi.z);		  
		  }
		  {
			  volCor_mask.mm2pixel(0, 0, 0, xyz.x, xyz.y, xyz.z);
			  volCor_mask.xyz2ijk(xyz.x, xyz.y, xyz.z, src_ijk_mask.x, src_ijk_mask.y, src_ijk_mask.z);
			  volCor_mask.mm2pixel(ray.d.x, ray.d.y, ray.d.z, xyz.x, xyz.y, xyz.z);
			  volCor_mask.xyz2ijk(xyz.x, xyz.y, xyz.z, cur_point_ijk.x, cur_point_ijk.y, cur_point_ijk.z);
			  vStep_mask = cur_point_ijk - src_ijk_mask;

			  volCor_mask.mm2pixel(ray_ori.x, ray_ori.y, ray_ori.z, xyz.x, xyz.y, xyz.z);
			  volCor_mask.xyz2ijk(xyz.x, xyz.y, xyz.z, src_ijk_mask.x, src_ijk_mask.y, src_ijk_mask.z);
		  }

		  for (t = tnear_mask; t <tnear_roi; t += tstep_mask){
			  //pos = ray_ori + ray.d*t;
			  //volCor_mask.mm2pixel(pos.x, pos.y, pos.z, xyz.x, xyz.y, xyz.z);
			  //volCor_mask.xyz2ijk(xyz.x, xyz.y, xyz.z, cur_point_ijk.x, cur_point_ijk.y, cur_point_ijk.z);
			  cur_point_ijk = src_ijk_mask + vStep_mask*t;
			  sum_mask += Text3DLinear(tex_vol_mask, cur_point_ijk);
		  }
		  if (bHit_roi){
			  for (int s=0;t<tfar_roi; t += tstep_roi, s++){
				  //pos = ray_ori + ray.d*t;
				  //volCor_roi.mm2pixel(pos.x, pos.y, pos.z, xyz.x, xyz.y, xyz.z);
				  //volCor_roi.xyz2ijk(xyz.x, xyz.y, xyz.z, cur_point_ijk.x, cur_point_ijk.y, cur_point_ijk.z);
				  cur_point_ijk = src_ijk_roi + vStep_roi*t;
				  val = Text3DLinear(tex_vol_roi, cur_point_ijk);
				  if (s == 0) sum_mask += val;
				  else        sum_roi  += val;
			  }
		  }
		  for (int s=0; t <tfar_mask; t += tstep_mask, s ++){
			  //pos = ray_ori + ray.d*t;
			  //volCor_mask.mm2pixel(pos.x, pos.y, pos.z, xyz.x, xyz.y, xyz.z);
			  //volCor_mask.xyz2ijk(xyz.x, xyz.y, xyz.z, cur_point_ijk.x, cur_point_ijk.y, cur_point_ijk.z);
			  cur_point_ijk = src_ijk_mask + vStep_mask*t;
			  val = Text3DLinear(tex_vol_mask, cur_point_ijk);
			  if (s == 0) sum_roi  += val;
			  else  	  sum_mask += val;
		  }
	  }

	  cuPointValue(dev_proj, pitch_dev_proj, para.nv, i, j) =  sum_mask*tstep_mask + sum_roi*tstep_roi;
  }

  //dev_proj[j*proj_dim.x + i] = (sum+(tfar-t+halfVStep)*sample) * tStep;
}

__global__ void kernel_forwardProject(float2   fCosSin,
									  float3   src_pos,
									  cuParameter para,	
									  cuVolumeGeo vol_geo_roi,
									  float*   dev_proj,
									  int      pitch_dev_proj											 								 						
									  )
{

	int i = blockIdx.x*blockDim.x + threadIdx.x;
	int j = blockIdx.y*blockDim.y + threadIdx.y;

	if (i >= para.nu || j >= para.nv)
		return;

	float det_yMM = para.SID - para.SOD;
	float src_yMM = -para.SOD; 
	//float tStep = MIN3(para.dx, para.dy, para.dz);    //mm unit
	float fsum;
	float3 rot_pt1_mm;
	// Setting ray origin
	cuRay ray;
	ray.o = src_pos;

	cuDetectorCoor<float> detectorCor(para.du, para.dv, para.nu, para.nv, para.offset_nu, para.offset_nv);
	cuVolumeCoor<float>   volCor_roi(vol_geo_roi.dx, vol_geo_roi.dy, vol_geo_roi.dz, vol_geo_roi.nx, vol_geo_roi.ny, vol_geo_roi.nz);		  
	{
#if 0
		rot_pt1_mm = RotDetTbl()(i, j);
#else
		//detector
		float u, v, uMM, vMM;
		detectorCor.ij2xy(i, j, u, v);
		detectorCor.pixel2mm(u, v, uMM, vMM);
		float3 pt1_mm = make_float3(uMM, det_yMM, vMM);
		cuRotate(float(0), float(0), fCosSin.x, fCosSin.y, pt1_mm.x, pt1_mm.y, rot_pt1_mm.x, rot_pt1_mm.y); 
		rot_pt1_mm.z = pt1_mm.z;
#endif
	}
	ray.d = cuNormize(rot_pt1_mm - src_pos); 

	{
		float sum_roi  = 0;
		float tnear_roi, tfar_roi, t;
		float tstep_roi = vol_geo_roi.tStep;
		bool bHit_roi = cuIntersectBox(ray, vol_geo_roi.boxmin, vol_geo_roi.boxmax, &tnear_roi, &tfar_roi);
		float3 ray_ori = ray.o;		  
		float3 pos;
		float3 xyz, cur_point_ijk;
		if (bHit_roi){
			float3 vStep;
			float3 src_ijk;
			{
				volCor_roi.mm2pixel(0, 0, 0, xyz.x, xyz.y, xyz.z);
				volCor_roi.xyz2ijk(xyz.x, xyz.y, xyz.z, src_ijk.x, src_ijk.y, src_ijk.z);
				volCor_roi.mm2pixel(ray.d.x, ray.d.y, ray.d.z, xyz.x, xyz.y, xyz.z);
				volCor_roi.xyz2ijk(xyz.x, xyz.y, xyz.z, cur_point_ijk.x, cur_point_ijk.y, cur_point_ijk.z);
				vStep = cur_point_ijk - src_ijk;

				volCor_roi.mm2pixel(ray_ori.x, ray_ori.y, ray_ori.z, xyz.x, xyz.y, xyz.z);
				volCor_roi.xyz2ijk(xyz.x, xyz.y, xyz.z, src_ijk.x, src_ijk.y, src_ijk.z);
			}
			for (t=tnear_roi;t<tfar_roi; t += tstep_roi){
#if 0
				pos = ray_ori + ray.d*t;
				volCor_roi.mm2pixel(pos.x, pos.y, pos.z, xyz.x, xyz.y, xyz.z);
				volCor_roi.xyz2ijk(xyz.x, xyz.y, xyz.z, cur_point_ijk.x, cur_point_ijk.y, cur_point_ijk.z);
#else
				cur_point_ijk = src_ijk + vStep*t;
#endif
				//sum_roi += Text3DLinear(tex_vol_roi, cur_point_ijk.x, cur_point_ijk.y, cur_point_ijk.z);
				sum_roi += Text3DLinear(tex_vol_roi, cur_point_ijk);
			}
		}

		cuPointValue(dev_proj, pitch_dev_proj, para.nv, i, j) = sum_roi*tstep_roi;
	}


  //dev_proj[j*proj_dim.x + i] = (sum+(tfar-t+halfVStep)*sample) * tStep;
}


__global__ void kernel_backProjection()
{

}

__global__ void kernel_Test1(cuDetectorCoor<float> detCoor)
{
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	int j = blockIdx.y*blockDim.y + threadIdx.y;
	float u, v;
	detCoor.mm2pixel(i, j, u, v);
}

void Test1()
{
	Parameter para;
	//cuDetectorCoor(T du, T dv, T nu, T nv, T offset_u, T offset_v)
	cuDetectorCoor<float> DetCoor(para.du, para.dv, para.nu, para.nv, para.offset_nu, para.offset_nv);
	kernel_Test1<<<128, 1>>>(DetCoor);
}

void Test2()
{
	Parameter para;
	double angle_rad = -(para.direction*para.vec_deg[0]/180.0*PI + para.volume_postion_angle);
	float2 fCosSin = make_float2(cos(angle_rad), sin(angle_rad));
	float3 src_pos;
	float*   dev_proj = NULL;
	cuParameter cupara(para);

	typedef float T;
	const float fcos = cos(angle_rad);
	const float fsin = sin(angle_rad);
	const cuVolumeCoor<float> volCor(para.dx, para.dy, para.dz, para.nx, para.ny, para.nz);
	const cuDetectorCoor<float> detectorCor(para.du, para.dv, para.nu, para.nv, para.offset_nu, para.offset_nv);
	const int size = para.nu*para.nv;
	const float det_yMM = para.SID - para.SOD;
	const float src_yMM = -para.SOD; 
	const Vector3<float> space(para.dx, para.dy, para.dz);
	const float tStep = 1; //T(space.Length());    //mm unit

	//volum range in pixel unit
	//object coordinate
	float3 boxmin, boxmax;
	{
		Vector3<float> box_min_ijk(0, para.ny-1, para.nz-1);
		Vector3<float> box_max_ijk(para.nx-1, 0, 0);
		volCor.ijk2xyz(box_min_ijk.x, box_min_ijk.y, box_min_ijk.z, boxmin.x, boxmin.y, boxmin.z);
		volCor.ijk2xyz(box_max_ijk.x, box_max_ijk.y, box_max_ijk.z, boxmax.x, boxmax.y, boxmax.z);
		volCor.pixel2mm(boxmin.x, boxmin.y, boxmin.z, boxmin.x, boxmin.y, boxmin.z);
		volCor.pixel2mm(boxmax.x, boxmax.y, boxmax.z, boxmax.x, boxmax.y, boxmax.z);
	}
	float3 rot_pt0_mm;
	{
		Vector3<T> pt0_mm(0, src_yMM, 0);
		::Rotate(T(0), T(0), fcos, fsin, pt0_mm.x, pt0_mm.y, rot_pt0_mm.x, rot_pt0_mm.y); 
		rot_pt0_mm.z = pt0_mm.z;
	}
	//cupara.boxmin = boxmin;
	//cupara.boxmax = boxmax;
	src_pos = rot_pt0_mm;
	cuVolumeGeo volGeo0(512, 512, 512, 0.157, 0.157, 0.146);
	cuVolumeGeo volGeo1(volGeo0.nx/2, volGeo0.ny/2, volGeo0.nz/2, volGeo0.dx*4, volGeo0.dy*4, volGeo0.dz*4);
	//float2   fCosSin, float3   src_pos, cuParameter para, cuVolumeGeo vol_geo_roi, cuVolumeGeo vol_geo_mask, float*   dev_proj, int      pitch_dev_proj,	
	//kernel_forwardProject<<<1024, 1024>>>(fCosSin, src_pos, NULL, cupara, volGeo0, volGeo1);
	DevData<float> devBuf(para.nu, para.nv);
	DevData<float>::DataRef devBufRef = devBuf.GetDataRef();

	DevArray3D<float> devArr0(volGeo0.nx, volGeo0.ny, volGeo0.nz);
	DevArray3D<float> devArr1(volGeo1.nx, volGeo1.ny, volGeo1.nz);

	devArr0.SetValue(1);
	devArr1.SetValue(2);

	devArr0.BindToTexture(&tex_vol_roi);
	devArr1.BindToTexture(&tex_vol_mask);

	kernel_forwardProject<<<1024, 1024>>>(fCosSin, src_pos, cupara, volGeo0, volGeo1, devBufRef.data, devBufRef.data_pitch);
}


template<typename T> inline
bool ForwardProjection(const Parameter& para, const double angle_rad, const VolumeT<T>& volume, DevData<T>& image)
{
	cuParameter cupara(para);
	float2 fCosSin = make_float2(cos(angle_rad), sin(angle_rad));
	float3 src_pos;
	float*   dev_proj = NULL;
	const cuVolumeCoor<float> volCor(para.dx, para.dy, para.dz, para.nx, para.ny, para.nz);
	const cuDetectorCoor<float> detectorCor(para.du, para.dv, para.nu, para.nv, para.offset_nu, para.offset_nv);
	const int size = para.nu*para.nv;
	const float det_yMM = para.SID - para.SOD;
	const float src_yMM = -para.SOD; 
	//volum range in pixel unit
	//object coordinate
	float3 rot_pt0_mm;
	{
		Vector3<T> pt0_mm(0, src_yMM, 0);
		::Rotate(T(0), T(0), fCosSin.x, fCosSin.y, pt0_mm.x, pt0_mm.y, rot_pt0_mm.x, rot_pt0_mm.y); 
		rot_pt0_mm.z = pt0_mm.z;
	}

	VERIFY_TRUE(volume.dx > 0 && volume.dy > 0 && volume.dz > 0);
	cuVolumeGeo volGeoRoi(volume.nx, volume.ny, volume.nz, volume.dx, volume.dy, volume.dz);
	//if (bWithMask){
	//}else
	{
		dim3 blocks(16, 16);
		dim3 grids(UpDivide(para.nu, blocks.x), UpDivide(para.nv, blocks.y));
		kernel_forwardProject<<<grids, blocks>>>(fCosSin, rot_pt0_mm, cupara, volGeoRoi, image.GetData(), image.DataPitch());
		cudaDeviceSynchronize();
		CUDA_CHECK_ERROR;
	}
	return true;
}

template<typename T> inline
	bool cudaCTForwardProjection(const ProjectionImageT<T>& proj, const Parameter& para, const Volume& volume, DevArray3D<T>* pDevVolRoi, DevArray3D<T>* pDevVolMask, ImageT<T>& forward_proj, DevData<T>& dev_forward_proj, std::vector<DevData<T>>* pvec_dev_forward_proj, ProjectionImageT<T>* pforward_proj_2th=NULL, int proj_index = -1)
{
	bool bRtn = true;
	try{
		//if (proj_index == -1) DestroyAllWindow();

		//DevArray3D<T> devArr0(volume.nx, volume.ny, volume.nz);
		//devArr0.CopyFromHost(volume.buffer, volume.nx, volume.nx, volume.ny, volume.nz);
		//devArr0.BindToTexture(tex_vol_roi);

		if (pDevVolRoi){
			pDevVolRoi->BindToTexture(&tex_vol_roi);
		}
		if (pDevVolMask){
			pDevVolMask->BindToTexture(&tex_vol_mask);
		}

		//ImageT<T>& image = forward_proj;
		ImageT<T>& image = forward_proj;
		VERIFY_TRUE(image.width > 0 && image.height > 0);
		//DevData<T>& devBuf = dev_forward_proj;
		//devBuf.MallocBuffer(para.nu, para.nv) ;

		int i = 0;
		const int width = para.nu;
		const int height = para.nv;
		//ImageT<T> image(para.nu, para.nv);
		int nProjCount = para.nProj;
		if (proj_index >= 0){
			VERIFY_TRUE(dev_forward_proj.width > 0 && dev_forward_proj.height > 0);
			i = proj_index;
			nProjCount = MIN(proj_index+1, para.nProj);
		}else{
			if (pvec_dev_forward_proj){
				VERIFY_TRUE(!pvec_dev_forward_proj->empty());
				VERIFY_TRUE((*pvec_dev_forward_proj)[0].channels >= 2);
				VERIFY_TRUE((*pvec_dev_forward_proj).size() >= nProjCount);
			}
		}
		DWORD total_time = 0;
		for (; i<nProjCount; i += para.step){
			image.Clear();
			FrameT<T> frame;
			if (proj.GetCount() > 0) {
				if (!proj.GetFrame(frame, i)) 
					break;
			}

			//VERIFY_TRUE(para.vec_rot_det_xy_tbl [i].size() == para.nu);
			//VERIFY_TRUE(para.vec_rot_det_z_tbl.size() == para.nv);
			//LoadConstant(c_rot_det_xy_tbl, &para.vec_rot_det_xy_tbl[i][0], para.nu);
			//LoadConstant(c_rot_det_z_tbl, &para.vec_rot_det_z_tbl[0], para.nv);
			double angle_rad = -(para.direction*para.vec_deg[i]/180.0*PI + para.volume_postion_angle);


			DWORD tm = timeGetTime();
			//bRtn = ForwardProjection(para, angle_rad, volume, frame, image);
			if (proj_index >= 0){
				bRtn = ForwardProjection(para, angle_rad, volume, dev_forward_proj);
				dev_forward_proj.CopyToHost(image.buffer, image.width, image.width, image.height);
			}else{
				bool bFwdProj = false;
				if (pvec_dev_forward_proj){
					bRtn = ForwardProjection(para, angle_rad, volume, (*pvec_dev_forward_proj)[i]);
					(*pvec_dev_forward_proj)[i].CopyToHost(image.buffer, image.width, image.width, image.height);
					bFwdProj = true;
				}
				if (pforward_proj_2th){
					if (!bFwdProj) bRtn = ForwardProjection(para, angle_rad, volume, dev_forward_proj);
					dev_forward_proj.CopyToHost(image.buffer, image.width, image.width, image.height);
					pforward_proj_2th->SetFrame(image.buffer, image.width, image.height, i);
				}
			}

			tm = timeGetTime() - tm;
			total_time += tm;

			if (i != 0) ClearCurrentLine();
			printf("ForwProj:%03d/%03d, angle=%03.2f,", i, para.nProj, angle_rad*180.0/PI);
			printf("time=%5d ms/%.3f m, ", tm, total_time/(1000*60.f));
			
			if (!frame.buffer)
				continue;
			{
				char szPath[1024] = "";
				sprintf(szPath, "%s%s/img%04d.raw", para.str_working_folder.c_str(), "proj-img", i);
				//	::WriteToFile<T, T>(szPath, image.buffer, width*height);
			}
			assert(bRtn);
			ImageT<T> _frame(frame.buffer, width, height);
		
			ImageT<T> img_scale(image.buffer, width, height);
			T a, b;
			::LinearLeastMeanSquare(image.buffer, _frame.buffer, width*height, a, b);
			img_scale *= a;
			img_scale += b;
			printf("LMS(a=%.3f,b=%.3f)", a, b);
			ImageT<T> diff = img_scale - _frame;

			if ( (i+1)%32 != 0) continue;
			diff.display("diff_ori_and_Forward_Projection");
			frame.display("original_raw_projection");
			image.display("volume_Forward_Projection");
		}
		printf("\n");
	}catch(std::string& error){
		printf("error : %s\n", error.c_str());
		bRtn = false;
	}catch(...){
		printf("error: %s\n", __FUNCTION__);
		bRtn = false;
	}
	return bRtn;
}


template<typename T> inline
bool cudaCTForwardProjection(const ProjectionImageT<T>& proj, const Parameter& para, const Volume& volume, ProjectionImageT<T>& proj_forward, int proj_index = -1)
{
	bool bRtn = true;
	try{
		if (proj_index == -1)
			DestroyAllWindow();

		DevArray3D<T> devArr0(volume.nx, volume.ny, volume.nz, cudaArraySurfaceLoadStore);
		devArr0.CopyFromHost(volume.buffer, volume.nx, volume.nx, volume.ny, volume.nz);
		devArr0.BindToTexture(&tex_vol_roi);

		DevData<T> devBuf(para.nu, para.nv);
		//DevData<T>::DataRef devBufRef = devBuf.GetDataRef();

		int i = 0;
		const int width = para.nu;
		const int height = para.nv;
		ImageT<T> image(para.nu, para.nv);
		int nProjCount = para.nProj;
		if (proj_index >= 0){
			i = proj_index;
			nProjCount = MIN(proj_index+1, para.nProj);
		}

		for (; i<nProjCount; i += para.step){
			image.Clear();
			FrameT<T> frame;
			if (!proj.GetFrame(frame, i)) break;
			double angle_rad = -(para.direction*para.vec_deg[i]/180.0*PI + para.volume_postion_angle);
			DWORD tm = timeGetTime();
			//bRtn = ForwardProjection(para, angle_rad, volume, frame, image);
			ForwardProjection(para, angle_rad, volume, devBuf);
			tm = timeGetTime() - tm;

			devBuf.CopyToHost(image.buffer, image.width, image.width, image.height);
			VERIFY_TRUE(proj_forward.SetFrame(image.buffer, width, height, i));
			if (i != 0) ClearCurrentLine();
			printf("ForwProj:%03d/%03d, angle=%03.2f,", i, para.nProj, angle_rad*180.0/PI);
			printf("time=%5d ms, ", tm);
			frame.display("original_raw_projection");
			{
				char szPath[1024] = "";
				sprintf(szPath, "%s%s/img%04d.raw", para.str_working_folder.c_str(), "proj-img", i);
				//	::WriteToFile<T, T>(szPath, image.buffer, width*height);
			}
			assert(bRtn);
			image.display("volume_Forward_Projection");
			ImageT<T> _frame(frame.buffer, width, height);
			ImageT<T> frame_smooth(frame.buffer, width, height);
			{
				ImageT<T> img_scale(image.buffer, width, height);
				T a, b;
				::LinearLeastMeanSquare(image.buffer, _frame.buffer, width*height, a, b);
				img_scale *= a;
				img_scale += b;
				printf("LMS(a=%.3f,b=%.3f)", a, b);
				ImageT<T> diff = img_scale - _frame;
				diff.display("diff_ori_and_Forward_Projection");
			}
		}
		//printf("\n");
	}catch(std::string& error){
		printf("error : %s\n", error.c_str());
		bRtn = false;
	}catch(...){
		printf("error: %s\n", __FUNCTION__);
		bRtn = false;
	}
	return bRtn;
}


bool cudaCTForwardProjection_f32(const ProjectionImageT<float>& proj, const Parameter& para, const Volume& volume, DevArray3D<float>* pDevVolRoi, DevArray3D<float>* pDevVolMask, ImageT<float>& forward_proj, DevData<float>& dev_forward_proj, std::vector<DevData<float>>* pvec_dev_forward_proj, ProjectionImageT<float>* pforward_proj_2th/*=NULL*/, int proj_index/* = -1*/){
	return cudaCTForwardProjection(proj, para, volume, pDevVolRoi, pDevVolMask, forward_proj, dev_forward_proj, pvec_dev_forward_proj, pforward_proj_2th, proj_index);
}

void TestForwardProjection0()
{
	bool bRtn;
	Parameter para;
	para.nx = para.ny = para.nz = 512;
	para.dx = para.dy = 0.157;
	para.dz = 0.146;
	para.Display();
	ProjectionImageT<float> proj, proj_2th;
	//VERIFY_TRUE(proj.LoadFolder<float>(para.str_proj_img_folder.c_str(), para.nu, para.nv, para.nProj, para.step));
	VERIFY_TRUE(proj.MallocBuffer(para.nu, para.nv, para.nProj, para.step).IsValid());
	proj_2th.MallocBuffer(para.nu, para.nv, para.nProj, para.step);
	Volume volum(para.nx, para.ny, para.nz);
	//char path[1024] = "C:/threshold.volume";
	//char path[1024] = "C:/temp/3dCT-101/volume/proj_512-general.volume";
	char path[1024] = "C:/temp/3dCT-502/proj-512.volume";
	VERIFY_TRUE(volum.Load(path));
	volum.SetUint(para.dx, para.dy, para.dz);

	//Rect32i roi(4, 2, para.nu-4*2, para.nv-2*2);

	//proj.SetRoi(roi);
	//CTBackProjection<float>(proj, para, volum);

	cudaCTForwardProjection(proj, para, volum, proj_2th);
}

void TestForwardProjection1()
{
//bool CTForwardProjection(const ProjectionImageT<T>& proj, std::vector<DevData<T>>& dev_proj, const Parameter& para, 
//	const Volume& volume, DevArray3D<T>* pDevVolRoi, DevArray3D<T>* pDevVolMask, ImageT<T>& forward_proj, DevData<T>& dev_forward_proj, int proj_index = -1)

	DISPLAY_FUNCTION;
	char szPath[256]; 
	//////////////////////////////////////////////////
	Parameter para;
	para.nx = para.ny = para.nz = 512;
	para.dx = para.dy = 0.175;
	para.dz = 0.146;
	para.Display();
	ProjectionImageT<float> proj;
	Volume volum(para.nx, para.ny, para.nz);
	proj.LoadFolder<float>(para.str_proj_img_folder.c_str(), para.nu, para.nv, para.nProj, para.step);
	proj.ShiftToNoneNegative();

	std::vector<DevData<float>> vec_dev_proj(para.nProj), vec_dev_forward_proj(para.nProj);
	for (int i=0; i<para.nProj; i += para.step){
		vec_dev_forward_proj[i].MallocBuffer(para.nu, para.nv);
		vec_dev_proj[i].MallocBuffer(para.nu, para.nv);
		FrameT<float> frame;
		proj.GetFrame(frame, i);
		vec_dev_proj[i].CopyFromHost(frame.buffer, frame.width, frame.width, frame.height, 1);
	}
	cudaDeviceSynchronize();

	ImageT<float> image(para.nu, para.nv);
	DevData<float> dev_image(para.nu, para.nv);
	//char path[1024] = "C:/threshold.volume";
	//char path[1024] = "C:/temp/3dCT-101/volume/proj_512-general.volume";
	char path[1024] = "C:/temp/3dCT-502/proj-512.volume";
	VERIFY_TRUE(volum.Load(path));
	volum.SetUint(para.dx, para.dy, para.dz);
	DevArray3D<float> dev_vol(volum.nx, volum.ny, volum.nz/*, cudaArraySurfaceLoadStore*/);
	dev_vol.CopyFromHost(volum.buffer, volum.nx, volum.nx, volum.ny, volum.nz);

	cudaCTForwardProjection_f32(proj, para, volum, &dev_vol, NULL, image, dev_image, &vec_dev_forward_proj, NULL, -1);
}

static int TestMain(int argc, char* argv[])
{
	int index = 0;
	if (argc > 1) index = atoi(argv[1]);
	switch (index)
	{
	case 0:
		TestForwardProjection0();
		break;
	case 1:
		TestForwardProjection1();
		break;
	default:
		TestForwardProjection0();
		break;
	}
	//Test2();
	//Test1();
	//TestIntersectBox();
	return 0;
}

int mainFP(int argc, char* argv[])
{
	TestMain(argc, argv);
	return 0;
}