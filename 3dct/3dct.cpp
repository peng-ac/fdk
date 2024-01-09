#include "../common/type.h"

template<typename T>
	bool BackProjection(const FrameT<T>& _frame, const Parameter& para, double angle_rad, Volume& vol, VolumeT<ushort>& volSumCount)
	{
		bool brtn = true;
		int m(0), n(0);
		const int width = _frame.width;
		const int height = _frame.height;
		const Rect32i* roi = _frame.roi;

		const T fcos = cos(angle_rad);
		const T fsin = sin(angle_rad);
		const VolumeCoor<T> volCor(para.dx, para.dy, para.dz, para.nx, para.ny, para.nz);
		const DetectorCoor<T> detectorCor(para.du,
			para.dv,
			para.nu,
			para.nv, 
			para.offset_nu, 
			para.offset_nv);

		ImageT<T> img_filted(_frame);

		_frame.display("frame");
		//WaitKey();
	
		if(para.str_filter == "none") {
#if 0
			img_filted *= para.cos_image_table.cos_sigma;
#else
			img_filted /= float(para.nProj) / para.step;
#pragma message("do not correct projection image by cos")
#endif
			//img_filted.display("img_filted filter_none", roi);
		}else {
			//std::vector<double> filt = filter(para.str_filter, width, 1);
#if 1
					ImageT<T> img_corrected(_frame);
			img_corrected *= para.cos_image_table.cos_sigma;
			//img_corrected *= para.cos_image_table.cos_sigma;
			//img_corrected *= para.cos_image_table.csc_sigma;
#if 1
					img_corrected *= para.AngularAndrampFactor;
#pragma message("use AngularAndrampFactor")
#else
#endif
			img_corrected.display("img_corrected");
			//DWORD tm = timeGetTime();
			filterImageRow(para.fft_kernel_real, img_corrected, img_filted);
			//char szPath[1024] = "";
			//static int idx = 0;
			//sprintf(szPath, "c:/proj_img_filtered/img%04d.raw", idx++);
			//img_filted.Save(szPath);
			//tm = timeGetTime() - tm;
#else
#pragma message("do not correct projection image by cos")
					filterImageRow(para.fft_kernel_real, _frame, img_filted);
#endif
			img_filted.display("convolutionImageRow", roi);
		}

		const T weightB = para.SOD;
		if (roi) img_filted.SetRoi(*roi);
		const int szXY = para.nx*para.ny;
		const int szXYZ = para.nx*para.ny*para.nz;	
#if 1
#pragma message("use BackProjection SSE")
		const Matrix2dReal4x4_SSE mat_proj = CalculateProjMat(para, angle_rad);
		{
#ifndef _DEBUG
#pragma omp parallel for// private(i,j,k)
#endif
			for (int i = 0; i < para.nz; i++) {
				const int offset_z = szXY*i;
				Matrix2dReal1x4 mat_ijk(0, 0, i, 1), mat_xyz;
				for (int j = 0; j < para.ny; j++) {
					mat_ijk.data[1] = j;
					const int offset_y = offset_z + vol.nx*j;
					for (int k = 0; k < para.nx; k++) {
						mat_ijk.data[0] = k;
						const int offset = offset_y + k;
						//MatrixMulti(mat_proj, mat_ijk, mat_xyz);
						mat_proj.Multiple(mat_ijk, mat_xyz);
						//if (j == 0 && k == 0){
						//	std::cout<<"i="<<i<<" "<<mat_xyz.data[0]<<" "<<mat_xyz.data[1]<<" "<<mat_xyz.data[2]<<" "<<mat_xyz.data[0]<<std::endl;
						//}
						const T& disSO_mm = mat_xyz.data[2];
						const T disSO_sc = 1 / disSO_mm;
						mat_xyz.data[0] *= disSO_sc;
						mat_xyz.data[1] *= disSO_sc;
#if 1
						T& i_det = mat_xyz.data[0];
						T& j_det = mat_xyz.data[1];
						if (img_filted.IsValid(i_det, j_det)) {
							//weighted fbp
							vol.buffer[offset] += img_filted(i_det, j_det)*disSO_sc*disSO_sc;
							//vol.buffer[offset] += img_filted(i_det, j_det);
							volSumCount.buffer[offset]++;
						}
#else
#pragma message("do not use weight")
						//#pragma message("use nearest interploter")
						{
							T& i_det = mat_xyz.data[0];
							T& j_det = mat_xyz.data[1];
							if (img_filted.IsValid(i_det, j_det)) {
								//weighted fbp
								//vol.buffer[offset] += img_filted(i_det, j_det)*disSO_mm*weightB;
								vol.buffer[offset] += img_filted(i_det, j_det);
								volSumCount.buffer[offset]++;
							}
							else {
								const int xx[4] = { i_det - 0.5, i_det + 0.5, i_det - 0.5, i_det + 0.5 };
								const int yy[4] = { j_det - 0.5, j_det + 0.5, j_det + 0.5, j_det - 0.5 };
								for (int s = 0; s < 4; s++) {
									if (img_filted.IsValid(xx[s], yy[s])) {
										//weighted fbp
										//vol.buffer[offset] += img_filted(i_det, j_det)*disSO_mm*weightB;
										vol.buffer[offset] += img_filted(xx[s], yy[s]);
										volSumCount.buffer[offset]++;
										break;
									}
								}
							}
						}

						//{
						//	int i_det = mat_xyz.data[0]+0.5;
						//	int j_det = mat_xyz.data[1]+0.5;
						//	if (img_filted.IsValid(i_det, j_det)){
						//		//weighted fbp
						//		//vol.buffer[offset] += img_filted(i_det, j_det)*disSO_mm*weightB;
						//		vol.buffer[offset] += img_filted(i_det, j_det);
						//		volSumCount.buffer[offset] ++;
						//	}else{
						//		i_det = mat_xyz.data[0]-0.5;
						//		j_det = mat_xyz.data[1]-0.5;
						//		if (img_filted.IsValid(i_det, j_det)){
						//			//weighted fbp
						//			//vol.buffer[offset] += img_filted(i_det, j_det)*disSO_mm*weightB;
						//			vol.buffer[offset] += img_filted(i_det, j_det);
						//			volSumCount.buffer[offset] ++;
						//		}else{
						//			i_det = mat_xyz.data[0]+0.5;
						//			j_det = mat_xyz.data[1]-0.5;
						//			if (img_filted.IsValid(i_det, j_det)){
						//				//weighted fbp
						//				//vol.buffer[offset] += img_filted(i_det, j_det)*disSO_mm*weightB;
						//				vol.buffer[offset] += img_filted(i_det, j_det);
						//				volSumCount.buffer[offset] ++;
						//			}else{
						//				i_det = mat_xyz.data[0]-0.5;
						//				j_det = mat_xyz.data[1]+0.5;
						//				if (img_filted.IsValid(i_det, j_det)){
						//					//weighted fbp
						//					//vol.buffer[offset] += img_filted(i_det, j_det)*disSO_mm*weightB;
						//					vol.buffer[offset] += img_filted(i_det, j_det);
						//					volSumCount.buffer[offset] ++;
						//				}

						//			}
						//		}
						//	}
						//}
#endif
					}
				}
			}

			return true;
		}
#else
#pragma message("use BackProjection C")

		//	#ifndef _DEBUG
#pragma omp parallel for// private(i,j,k)
			//	#endif
				for(int i = 0 ; i < para.nz ; i++) {
			const int offset_z = szXY*i;
			Matrix2dReal1x4 mat_ijk(0, 0, i, 1), mat_xyz;
			for (int j = 0; j < para.ny; j++) {
				mat_ijk.data[1] = j;
				const int offset_y = offset_z + vol.nx*j;
				for (int k = 0; k < para.nx; k++) {
					mat_ijk.data[0] = k;
					//Matrix2dReal1x4 xyz;
					const int offset = offset_y + k;
					T xmm_vol, ymm_vol, zmm_vol;
					T i_det, j_det;
					T scale;
					T disSO;
					{
						T x(0), y(0), z(0), rx(0), ry(0), rz(0);
						volCor.ijk2xyz(k, j, i, x, y, z);
						//MatrixMulti(mat0, mat, xyz); 

						Rotate<T, T, T, T>(0, 0, fcos, fsin, x, y, rx, ry);
						rz = z;
						volCor.pixel2mm(rx, ry, rz, xmm_vol, ymm_vol, zmm_vol);

						//MatrixMulti(mat1, xyz, xyz); 
						//MatrixMulti(mat2, xyz, xyz); 
						//MatrixMulti(mat3, xyz, xyz); 
					}
					{	
						T umm_det(0), vmm_det(0);
						{
							disSO = para.SOD + ymm_vol;
							scale = para.SID / disSO;
							umm_det = scale*xmm_vol;
							vmm_det = scale*zmm_vol;
						}
						T u_pix(0), v_pix(0);
						detectorCor.mm2pixel(umm_det, vmm_det, u_pix, v_pix);
						//MatrixMulti(mat4, xyz, xyz); 
						//MatrixMulti(mat5, xyz, xyz); 

						detectorCor.xy2ij(u_pix, v_pix, i_det, j_det);

					}
						
					//MatrixMulti(mat_proj, mat_ijk, mat_xyz);
					//mat_proj.Multiple(mat_ijk, mat_xyz);
					//Vector3<float> tmp(mat_xyz.data[0]/mat_xyz.data[2], mat_xyz.data[1]/mat_xyz.data[2], 1);

					if(img_filted.IsValid(i_det, j_det)) {
						//weighted fbp
						//vol.buffer[offset] += img_filted(i_det, j_det)/disSO*weightB;
						if(para.str_filter == "none") {
							vol.buffer[offset] += img_filted(i_det, j_det);
						}else {
							vol.buffer[offset] += img_filted(i_det, j_det)*std::pow(weightB / disSO, 2);
						}
						volSumCount.buffer[offset]++;
					}
				}
			}
		}
#endif
		return brtn;
	}


template<typename T>
	bool CTBackProjection(ProjectionImageT<T>& proj, const Parameter& para, Volume& img, int proj_index = -1)
	{
		//extern bool cudaCTBackprojectionReal32(ProjectionImage& proj, const Parameter& para, Volume& img);
		//return cudaCTBackprojectionReal32(proj, para, img);
		bool bRtn = true;
		if (proj_index == -1) {
			DestroyAllWindow();
			img.Zeros();
		}
		Volume vol(img.nx, img.ny, img.nz);
		VolumeT<ushort> volSumCount(img.nx, img.ny, img.nz);
		DWORD total_time = 0;

		int nStart = 0;
		//Rect32i rcSubImg(96, 96, para.nu-96*2, para.nv-96*2);
		int step = para.step;
		int nProjCount = para.nProj;
		if (proj_index >= 0) {
			nStart = proj_index;
			nProjCount = 1;
		}
		for (int idx = 0; idx < nProjCount; idx += step) {
			int i = idx + nStart;
			if (i >= para.nProj) i -= para.nProj;
			const int& frame_index = i;
			FrameT<T> frame;
			bRtn = proj.GetFrame(frame, frame_index);
			VERIFY_TRUE(bRtn);
			if (!bRtn) break;	
			//frame.display("sub_frame_96-96", &rcSubImg);
			double angle_rad = para.direction*para.vec_deg[frame_index] / 180.0*PI + para.volume_postion_angle;
			DWORD tm = timeGetTime();
			bRtn = BackProjection<T>(frame, para, angle_rad, img, volSumCount);
			VERIFY_TRUE(bRtn);
			if (!bRtn) break;
			tm = timeGetTime() - tm;
			total_time += tm;
			//if (i != 0) ClearCurrentLine();
			printf("CTBackProjection(%03d/%03d=%.1f%s, time=%d ms/%.3f m, roi(x(%d) y(%d))\n",
				i,
				para.nProj,
				float(i + 1)/para.nProj*100,
				"%",
				tm,
				float(total_time)/(60 * 1000),
				proj.roi?proj.roi->x:0,
				proj.roi?proj.roi->y:0);
			//img += vol;

			vol = img;
			//vol /= i+1.0;
			//if (para.str_filter != "none")
#if 0
					vol /= volSumCount;
#else
#pragma message("volume data without weight")
#endif

#if 0
			for (int i = 0; i < para.ny; i++) {
				vol.SliceX(i);
				vol.SliceY(i);
				vol.SliceZ(i);
			}
#else
			vol.SliceX(img.nx / 8);
			vol.SliceY(img.ny / 8);
			vol.SliceZ(img.nz / 2);
#endif
		}
		printf("\n");
		//if (para.str_filter != "none")
		if(bRtn) {
			//img *= T(para.nProj)/step;
#if 0
					img *= (para.nProj / step);
#if 0
			img /= volSumCount;
#else
#pragma message("do not use average volume")
#endif
#endif
			//img *= float(para.nProj)/float(para.step);
		}

		//char szPath[1024];
		//sprintf(szPath, "D:/CT_Scanner/volume/proj_%d.volume", i);
		//sprintf(szPath, "c:/temp/volume/proj_%d.volume", i);
		//sprintf(szPath, "%s%s/proj_%d.volume", para.str_working_folder.c_str(), "volume", para.nProj);
		//img.Save(szPath);

		return bRtn;
	}

template<typename T>
	bool CTBackProjection(ProjectionImageT<T>& proj, ProjectionImageT<T>& proj_mask, const Parameter& para, Volume& img)
	{
		//extern bool cudaCTBackprojectionReal32(ProjectionImage& proj, const Parameter& para, Volume& img);
		//return cudaCTBackprojectionReal32(proj, para, img);
		bool bRtn = true;
		DestroyAllWindow();

		img.Zeros();
		Volume vol(img.nx, img.ny, img.nz);
		VolumeT<ushort> volSumCount(img.nx, img.ny, img.nz);
		DWORD total_time = 0;

		int nStart = 0;
		Rect32i rcSubImg(96, 96, para.nu - 96 * 2, para.nv - 96 * 2);
		int step = para.step;
		for (int idx = 0; idx < para.nProj; idx += step) {
			int i = idx + nStart;
			if (i >= para.nProj) i -= para.nProj;
			const int& frame_index = i;
			FrameT<T> frame, mask;
			bRtn = proj.GetFrame(frame, frame_index);
			VERIFY_TRUE(bRtn);
			bRtn = proj_mask.GetFrame(mask, frame_index);
			VERIFY_TRUE(bRtn);
			if (!bRtn) break;	
			frame.display("sub_frame_96-96", &rcSubImg);
			double angle_rad = para.direction*para.vec_deg[frame_index] / 180.0*PI + para.volume_postion_angle;
			DWORD tm = timeGetTime();
			bRtn = BackProjection<T>(frame, mask, para, angle_rad, img, volSumCount);
			VERIFY_TRUE(bRtn);
			if (!bRtn) break;
			tm = timeGetTime() - tm;
			total_time += tm;
			if (i != 0) ClearCurrentLine();
			printf("CTBackProjection(%03d/%03d=%.1f%s, time=%d ms/%.3f m, roi(x(%d) y(%d))",
				i,
				para.nProj,
				float(i + 1)/para.nProj*100,
				"%",
				tm,
				float(total_time)/(60 * 1000),
				proj.roi?proj.roi->x:0,
				proj.roi?proj.roi->y:0);
			//img += vol;

			vol = img;
			//vol /= i+1.0;
			//if (para.str_filter != "none")
#if 1
					vol /= volSumCount;
#else
#pragma message("volume data without weight")
#endif

#if 0
			for (int i = 0; i < para.ny; i++) {
				vol.SliceX(i);
				vol.SliceY(i);
				vol.SliceZ(i);
			}
#else
			vol.SliceX(img.nx / 8);
			vol.SliceY(img.ny / 8);
			vol.SliceZ(img.nz / 2);
#endif
		}
		printf("\n");
		//if (para.str_filter != "none")
		if(bRtn) {
			//img *= T(para.nProj)/step;
#if 1
					img *= (para.nProj / step);
#if 0
			img /= volSumCount;
#else
#pragma message("do not use average volume")
#endif
#endif
			//img *= float(para.nProj)/float(para.step);
		}

		//char szPath[1024];
		//sprintf(szPath, "D:/CT_Scanner/volume/proj_%d.volume", i);
		//sprintf(szPath, "c:/temp/volume/proj_%d.volume", i);
		//sprintf(szPath, "%s%s/proj_%d.volume", para.str_working_folder.c_str(), "volume", para.nProj);
		//img.Save(szPath);

		return bRtn;
	}

template<typename T>
	bool ForwardProjection(const Parameter& para, double angle_rad, const Volume& volum, const FrameT<T>& frame, ImageT<T>& image)
	{
		//std::cout << angle_rad << std::endl;
		bool bRtn = true;
		assert(image.width == para.nu && image.height == para.nv);
		const T fcos = cos(angle_rad);
		const T fsin = sin(angle_rad);
		const VolumeCoor<T> volCor(para.dx, para.dy, para.dz, para.nx, para.ny, para.nz);
		const DetectorCoor<T> detectorCor(para.du, para.dv, para.nu, para.nv, para.offset_nu, para.offset_nv);
		const int size = para.nu*para.nv;
		const T det_yMM = para.SID - para.SOD;
		const T src_yMM = -para.SOD; 
		const Vector3<T> space(para.dx, para.dy, para.dz);
		const T tStep = T(MIN3(para.dx, para.dy, para.dz));     //mm unit

		//volum range in pixel unit
		//object coordinate
		Vector3<T> boxmin, boxmax;
		{
			Vector3<T> box_min_ijk(0, para.ny - 1, para.nz - 1);
			Vector3<T> box_max_ijk(para.nx - 1, 0, 0);
			volCor.ijk2xyz(box_min_ijk.x, box_min_ijk.y, box_min_ijk.z, boxmin.x, boxmin.y, boxmin.z);
			volCor.ijk2xyz(box_max_ijk.x, box_max_ijk.y, box_max_ijk.z, boxmax.x, boxmax.y, boxmax.z);
			volCor.pixel2mm(boxmin.x, boxmin.y, boxmin.z, boxmin.x, boxmin.y, boxmin.z);
			volCor.pixel2mm(boxmax.x, boxmax.y, boxmax.z, boxmax.x, boxmax.y, boxmax.z);
		}
		//beam source
		Vector3<T> rot_pt0_mm;
		{
			Vector3<T> pt0_mm(0, src_yMM, 0);
			::Rotate(T(0), T(0), fcos, fsin, pt0_mm.x, pt0_mm.y, rot_pt0_mm.x, rot_pt0_mm.y); 
			rot_pt0_mm.z = pt0_mm.z;
		}
		//beam vector
		Vector3<T> normal_beam_vec(0, 1, 0);
		{
			volCor.pixel2mm(normal_beam_vec.x, normal_beam_vec.y, normal_beam_vec.z, normal_beam_vec.x, normal_beam_vec.y, normal_beam_vec.z);
			::Rotate(T(0), T(0), fcos, fsin, normal_beam_vec.x, normal_beam_vec.y, normal_beam_vec.x, normal_beam_vec.y); 
		}
		const T normal_beam_vec_length = normal_beam_vec.Length();

#ifndef _DEBUG 
#pragma omp parallel for
#endif
		for (int sz = 0; sz < size; sz++) {
			bool bHit = false;
			RayT<T>    ray;
			int i = sz % para.nu;     //x
			int j = sz / para.nu;     //y
			Vector3<T> rot_pt1_mm;
			T tnear = 0;
			T tfar = 0;
			//Vector3<T> step;
			int count = 0;
			{
				//detector
				T u, v, uMM, vMM;
				detectorCor.ij2xy(i, j, u, v);
				detectorCor.pixel2mm(u, v, uMM, vMM);
				Vector3<T> pt1_mm(uMM, det_yMM, vMM);
				::Rotate(T(0), T(0), fcos, fsin, pt1_mm.x, pt1_mm.y, rot_pt1_mm.x, rot_pt1_mm.y); 
				rot_pt1_mm.z = pt1_mm.z;
			}
			{
				ray.d = rot_pt1_mm - rot_pt0_mm;
				//volCor.mm2pixel(ray.d.x, ray.d.y, ray.d.z, ray.d.x, ray.d.y, ray.d.z);
				//volCor.mm2pixel(rot_pt0_mm.x, rot_pt0_mm.y, rot_pt0_mm.z, ray.o.x, ray.o.y, ray.o.z);
				ray.o = rot_pt0_mm;
				ray.d.Normize();
			}
			{
				bHit = IntersectBox(ray, boxmin, boxmax, &tnear, &tfar);
				if (!bHit) {
					//printf("IntersectBox false\n");
					continue;
				}
			}

			{
				//Vector3<T> dirInMM = space * ray.d;
				//T vStep = tStep / dirInMM.Length();
				//Vector3<T> step = ray.d * vStep;
				T vStep = tStep;
				Vector3<T> step = ray.d*vStep;
				T halfVStep = 0.5*vStep;
				//tnear += halfVStep;
				Vector3<T> pos = ray.o + ray.d*tnear;

				if (tnear > tfar) {
					//std::cout<<"tnear > tfar"<<std::endl;
					continue;
				}

				T t;
				int nTotal = 0;
				T sample = T(0);
				T sum    = T(0);
				count = (tfar - tnear) / vStep;
	
				Vector3<T> cur_point_xyz;
				Vector3<float> cur_point_ijk;

				//int cnt = 0;
				for(t = tnear ; t <= tfar ; t += vStep) {
					volCor.mm2pixel(pos.x, pos.y, pos.z, cur_point_xyz.x, cur_point_xyz.y, cur_point_xyz.z);
					volCor.xyz2ijk(cur_point_xyz.x, cur_point_xyz.y, cur_point_xyz.z, cur_point_ijk.x, cur_point_ijk.y, cur_point_ijk.z);
				
					sample = volum(cur_point_ijk.x, cur_point_ijk.y, cur_point_ijk.z);
					//if (volum.IsValid(cur_point_ijk.x, cur_point_ijk.y, cur_point_ijk.z)){
					//	sample = volum(cur_point_ijk.x, cur_point_ijk.y, cur_point_ijk.z);
					//}else{
					//	sample = volum(cur_point_ijk.x, cur_point_ijk.y, cur_point_ijk.z);
					//	//Vector3<int> cur_point_ijk_int(cur_point_ijk.x+0.5, cur_point_ijk.y+0.5, cur_point_ijk.z+0.5);
					//	//if (volum.IsValid(cur_point_ijk_int.x, cur_point_ijk_int.y, cur_point_ijk_int.z)){
					//	//	sample = volum(cur_point_ijk_int.x, cur_point_ijk_int.y, cur_point_ijk_int.z);
					//	//}else{
					//	//	sample = 0;
					//	//	VERIFY_TRUE(0);
					//	//	continue;
					//	//}
					//}
					//if (sample > 2.5) cnt ++;
					//if (cnt > 90) sample *= (0.2 + (0.4*rand()-RAND_MAX/2)/RAND_MAX);
					//if (cnt > 90){ sample = 0; break;}
					sum += sample;
					pos = pos + step;
					nTotal++;
				}
				//float weight = para.nx/(tfar - tnear);
				//image(i, j) = (sum + (tfar-t+halfVStep)*sample)*tStep;	
				// 1 mm vector's sample value
				//image(i, j) = (sum + (tfar-t+halfVStep)/vStep*sample)*tStep;//*weight;	
				image(i, j) = sum *tStep;
			}
		}

#if 0
		image *= para.cos_image_table.csc_sigma;
#else
#pragma message("do not use cos image correction")
#endif
		return bRtn;
	}

template<typename T>
	bool CTForwardProjection(const Parameter& para, const Volume& volume, ProjectionImageT<T>& proj_forward, int proj_index = -1)
	{
		bool bRtn = true;
		try {
			if (proj_index == -1)
				DestroyAllWindow();

			int i = 0;
			const int width = para.nu;
			const int height = para.nv;
			ImageT<T> image(para.nu, para.nv);
			int nProjCount = para.nProj;
			if (proj_index >= 0) {
				i = proj_index;
				nProjCount = MIN(proj_index + 1, para.nProj);
			}
			ImageT<T> frm(para.nu, para.nv);
			for (; i < nProjCount; i += para.step) {
				image.Clear();
				double angle_rad = -(para.direction*para.vec_deg[i] / 180.0*PI + para.volume_postion_angle);
				DWORD tm = timeGetTime();

				bRtn = ForwardProjection(para, angle_rad, volume, frm.GetFrame(), image);

				tm = timeGetTime() - tm;
				VERIFY_TRUE(proj_forward.SetFrame(image.buffer, width, height, i));
				//if (i != 0) ClearCurrentLine();
				printf("\nForwProj:%03d/%03d, angle=%03.2f,", i, para.nProj, angle_rad * 180.0 / PI);
				printf("time=%5d ms, ", tm);
				assert(bRtn);
				image.display("volume_Forward_Projection");
			}
			printf("\n");
		}
		catch (std::string& error) {
			printf("error : %s\n", error.c_str());
			bRtn = false;
		}
		catch (...) {
			printf("error: %s\n", __FUNCTION__);
			bRtn = false;
		}
		return bRtn;
	}


template<typename T>
	bool CTForwardProjection(const ProjectionImageT<T>& proj, const Parameter& para, const Volume& volume, ProjectionImageT<T>& proj_forward, int proj_index = -1)
	{
		bool bRtn = true;
		try {
			if (proj_index == -1)
				DestroyAllWindow();

			int i = 0;
			const int width = para.nu;
			const int height = para.nv;
			ImageT<T> image(para.nu, para.nv);
			int nProjCount = para.nProj;
			if (proj_index >= 0) {
				i = proj_index;
				nProjCount = MIN(proj_index + 1, para.nProj);
			}
			for (; i < nProjCount; i += para.step) {
				image.Clear();
				FrameT<T> frame;
				if (!proj.GetFrame(frame, i)) break;
				double angle_rad = -(para.direction*para.vec_deg[i] / 180.0*PI + para.volume_postion_angle);
				DWORD tm = timeGetTime();

				bRtn = ForwardProjection(para, angle_rad, volume, frame, image);

				tm = timeGetTime() - tm;
				VERIFY_TRUE(proj_forward.SetFrame(image.buffer, width, height, i));
				if (i != 0) ClearCurrentLine();
				printf("ForwProj:%03d/%03d, angle=%03.2f,", i, para.nProj, angle_rad * 180.0 / PI);
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

				//{
				//	Rect32i roi(8, 8, frame.width-8*2, frame.height-8*2);
				//	ImageT<T> img_scale(image.buffer, frame.width, frame.height);
				//	T kernel[5] = {0.2, 0.2, 0.2, 0.2, 0.2};
				//	std::vector<T> vecKernel(kernel, kernel+sizeof(kernel)/sizeof(kernel[0]));
				//	convoluteImage(vecKernel, vecKernel, frame.data, frame_smooth.buffer, frame.width, frame.height);
				//	frame_smooth.display("frame_smooth", &roi);
				//	T a, b;
				//	::LinearLeastMeanSquare(image.buffer, frame_smooth.buffer, frame.width*frame.height, a, b);
				//	img_scale *= a;
				//	img_scale += b;
				//	ImageT<T> diff_smooth = img_scale - frame_smooth;
				//	diff_smooth.display("diff_ori_and_smooth_Forward_Projection", &roi);
				//}
				//WaitKey(10);
			}
			printf("\n");
		}
		catch (std::string& error) {
			printf("error : %s\n", error.c_str());
			bRtn = false;
		}
		catch (...) {
			printf("error: %s\n", __FUNCTION__);
			bRtn = false;
		}
		return bRtn;
	}

template<typename T>
	bool CTForwardProjection(const ProjectionImageT<T>& proj, const ProjectionImageT<T>& proj_mask, const Parameter& para, const Volume& volume, ProjectionImageT<T>& proj_forward)
	{
		bool bRtn = true;
		try {
			DestroyAllWindow();

			int i = 0;
			const int width = para.nu;
			const int height = para.nv;
			ImageT<T> image(para.nu, para.nv);
			for (i = 0; i < para.nProj; i += para.step) {
				image.Clear();
				FrameT<T> frame, frame_mask;
				if (!proj.GetFrame(frame, i)) break;
				if (!proj_mask.GetFrame(frame_mask, i)) break;
				double angle_rad = -(para.direction*para.vec_deg[i] / 180.0*PI + para.volume_postion_angle);
				DWORD tm = timeGetTime();

				bRtn = ForwardProjection(para, angle_rad, volume, frame, image);

				tm = timeGetTime() - tm;
				VERIFY_TRUE(proj_forward.SetFrame(image.buffer, width, height, i));
				if (i != 0) ClearCurrentLine();
				printf("ForwProj:%03d/%03d, angle=%03.2f,", i, para.nProj, angle_rad * 180.0 / PI);
				printf("time=%5d ms, ", tm);
				frame.display("original_raw_projection");
				{
					char szPath[1024] = "";
					sprintf(szPath, "%s%s/img%04d.raw", para.str_working_folder.c_str(), "proj-img", i);
					//	::WriteToFile<T, T>(szPath, image.buffer, width*height);
				}
				assert(bRtn);
				frame_mask.display("frame_mask");
				image.display("volume_Forward_Projection");
				ImageT<T> _frame(frame);
				ImageT<T> frame_smooth(frame);	
				ImageT<T> img_mask(frame_mask);
				img_mask.Dilate(2);
				{
					//T kernel[] = {0.1, 0.8, 0.1};
					//std::vector<T> vecKernel(kernel, kernel+sizeof(kernel)/sizeof(kernel[0]));
					//convoluteImage(vecKernel, vecKernel, frame.data, frame_smooth.buffer, width, height);
					::MedianFilter(frame.buffer, frame_smooth.buffer, width, height);


					std::vector<T> vecSrc, vecProj_2th;
					std::vector<int> vecHeightIndex;
					for (int m = 0; m < height; m++) {
						bool bMask = false;
						for (int n = 0; n < width; n++) {
							if (img_mask(n, m) != 0) {
								vecHeightIndex.push_back(m);
								break;
							}
						}
					}
					for (int k = 0; k < vecHeightIndex.size(); k++) {
						int m = vecHeightIndex[k];
						for (int n = 0; n < width; n++) {
							vecSrc.push_back(frame_smooth(n, m));
							vecProj_2th.push_back(image(n, m));
						}
					}	
					{
						T a, b;
						::LinearLeastMeanSquare(&vecProj_2th[0], &vecSrc[0], vecSrc.size(), a, b);
						printf(" A = %f, B = %f\n", a, b);
						ImageT<T> img_inpaint_row_part(frame);

						for (int k = 0; k < vecHeightIndex.size(); k++) {
							int m = vecHeightIndex[k];
							for (int n = 0; n < width; n++) {
								img_inpaint_row_part(n, m) = image(n, m)*a + b;
							}
						}

						img_inpaint_row_part.display("img_inpaint_row_part");

						VERIFY_TRUE(proj_forward.SetFrame(img_inpaint_row_part.buffer, width, height, i));
					}

					//ImageT<T> img_inpaint_part(frame);
					//if (!vecSrc.empty()){
					//	T a, b;
					//	::LinearLeastMeanSquare(&vecProj_2th[0], &vecSrc[0], vecSrc.size(), a, b);
					//	printf("\nA = %f, B = %f\n", a, b);


					//	for (int k=0; k<width*height; k ++){
					//		if (img_mask.buffer[k] != 0){
					//			T s = image.buffer[k]*a + b;
					//			T w = 0.8;
					//			img_inpaint_part.buffer[k] = s*w + (1-w)*img_inpaint_part.buffer[k];
					//		}
					//	}
					//	img_inpaint_part.display("img_inpaint_part");
					//}

				}
				//{
				//	ImageT<T> img_scale(image);
				//	ImageT<T> img_inpaint(frame);

				//	T a, b;
				//	::LinearLeastMeanSquare(image.buffer, _frame.buffer, width*height, a, b);
				//	img_scale *= a;
				//	img_scale += b;
				//	printf("LMS(a=%f,b=%f)", a, b);
				//	ImageT<T> diff = img_scale - _frame;
				//	diff.display("diff_ori_and_Forward_Projection");

				//	
				//	for (int k=0; k<width*height; k ++){
				//		if (img_mask.buffer[k] != 0){
				//			img_inpaint.buffer[k] = img_scale.buffer[k]*0.8;
				//			//img_inpaint.buffer[k] = (img_inpaint.buffer[k] + img_scale.buffer[k])/2;
				//		}
				//	}
				//	img_inpaint.display("img_inpaint");

				//	
				//	img_mask = frame_mask;
				//	img_mask.Dilate();
				//	ImageT<T> img_src_filter(image);
				//	ImageT<T> img_inpaint_filter(image);
				//	filterImageRow(para.fft_kernel_real, frame, img_src_filter);
				//	filterImageRow(para.fft_kernel_real, img_inpaint, img_inpaint_filter);
				//	ImageT<T> img_filter_inpaint_filter(img_src_filter);
				//	img_src_filter.display("img_src_filter");
				//	img_inpaint_filter.display("img_inpaint_filter");
				//	for (int k=0; k<width*height; k ++){
				//		if (img_mask.buffer[k] != 0){
				//			img_filter_inpaint_filter.buffer[k] = img_inpaint_filter.buffer[k]*0.8;
				//			//img_inpaint.buffer[k] = (img_inpaint.buffer[k] + img_scale.buffer[k])/2;
				//		}
				//	}
				//	img_filter_inpaint_filter.display("img_filter_inpaint_filter");
				//}

				//{
				//	Rect32i roi(8, 8, frame.width-8*2, frame.height-8*2);
				//	ImageT<T> img_scale(image.buffer, frame.width, frame.height);
				//	T kernel[5] = {0.2, 0.2, 0.2, 0.2, 0.2};
				//	std::vector<T> vecKernel(kernel, kernel+sizeof(kernel)/sizeof(kernel[0]));
				//	convoluteImage(vecKernel, vecKernel, frame.data, frame_smooth.buffer, frame.width, frame.height);
				//	frame_smooth.display("frame_smooth", &roi);
				//	T a, b;
				//	::LinearLeastMeanSquare(image.buffer, frame_smooth.buffer, frame.width*frame.height, a, b);
				//	img_scale *= a;
				//	img_scale += b;
				//	ImageT<T> diff_smooth = img_scale - frame_smooth;
				//	diff_smooth.display("diff_ori_and_smooth_Forward_Projection", &roi);
				//}
				//WaitKey(10);
			}
			printf("\n");
		}
		catch (std::string& error) {
			printf("error : %s\n", error.c_str());
			bRtn = false;
		}
		catch (...) {
			printf("error: %s\n", __FUNCTION__);
			bRtn = false;
		}
		return bRtn;
	}


void Test2dProjection()
{
	//char szFolder[] = "D:/CT_Scanner/ct-rec-3d/Low960_768/";	
	//char szFolder[] = "c:/temp/Low960_768/";
	//char szFolder[] = "D:/CT_Scanner/High1920_1536/";	
	//char szFolder[] = "c:/temp/proj-img/";
	//char szFolder[] = "D:/CT_Scanner/proj-img/";

	//////////////////////////////////////////////////
	bool bRtn;
	std::shared_ptr<Parameter> pPara;
	if (0) {
		pPara = std::make_shared<Parameter>(); Parameter& para = *pPara;
		para.du = para.nu*para.du / 2048; para.dv = para.nv*para.dv / 2048;
		para.nu = para.nv = 2048;
		para.nProj = 1200;		
	}
	if (0) {
		pPara = std::make_shared<Parameter>(); Parameter& para = *pPara;
		para.du = para.nu*para.du / 512; para.dv = para.nv*para.dv / 512;
		para.nu = para.nv = 512;
		para.nProj = 1200;		
	}
	if (0) {
		pPara = std::make_shared<Parameter>(); Parameter& para = *pPara;
		para.du = para.nu*para.du / 512; para.dv = para.nv*para.dv / 512;
		para.nu = para.nv = 1024;
		para.nProj = 1024;		
	}	
	if (1) {
		pPara = std::make_shared<Parameter>(); Parameter& para = *pPara;
		para.du = para.nu*para.du / 1024; para.dv = para.nv*para.dv / 1024;
		para.nu = para.nv = 1024;
		para.nProj = 1024;		
	}		
	if (0) {
		pPara = std::make_shared<Parameter>(); Parameter& para = *pPara;
		para.du = para.nu*para.du / 1024; para.dv = para.nv*para.dv / 1024;
		para.nu = para.nv = 1024;
		para.nProj = 1200;		
	}	
	if (0) {
		pPara = std::make_shared<Parameter>(); Parameter& para = *pPara;
		para.du = para.nu*para.du / 1024; para.dv = para.nv*para.dv / 1024;
		para.nu = para.nv = 1024;
		para.nProj = 1024;		
	}	
	if (0) {
		pPara = std::make_shared<Parameter>(); Parameter& para = *pPara;
		para.du = para.nu*para.du / 2048; para.dv = para.nv*para.dv / 2048;
		para.nu = para.nv = 2048;
		para.nProj = 2048;		
	}		
	if (1) {
		pPara = std::make_shared<Parameter>(); Parameter& para = *pPara;
		para.du = para.nu*para.du / 4096; para.dv = para.nv*para.dv / 4096;
		para.nu = para.nv = 4096;
		para.nProj = 2400;		
	}
	//const int NU = 1280; int NV = 1024;
	//para.du = para.du*para.nu/NU;  para.nu = NU;
	//para.dv = para.dv*para.nv/NV;  para.nv = NV;
	//para.dx = para.dx*para.nx/1024;  para.nx = 1024;
	//para.dy = para.dy*para.ny/1024;  para.ny = 1024;
	//para.dz = para.dz*para.nz/1024;  para.nz = 1024;
	//para.nProj = 1024;
	Parameter& para = *pPara;
	para.init();

	para.Display();
	ProjectionImageT<float> proj, proj_2th;
	//VERIFY_TRUE(proj.LoadFolder<float>(para.str_proj_img_folder.c_str(), para.nu, para.nv, para.nProj, para.step));
	//VERIFY_TRUE(proj.MallocBuffer(para.nu, para.nv, para.nProj, para.step));
	proj_2th.MallocBuffer(para.nu, para.nv, para.nProj, para.step);
	Volume volum(para.nx, para.ny, para.nz);
	
	//char path[1024] = "C:/threshold.volume";
	//char path[1024] = "C:/temp/3dCT-101/volume/proj_512-general.volume";
	//char path[1024] = "C:/temp/sim_rect1.volume";
	char path[1024] = "/home/pengdadi/projects/data/phantom3d/Shepp-Logan-512x512x512.vol";
	//char path[1024] = "/home/chen/projects/data/phantom3d/";
	//sprintf(path, "%s%s", "D:/CT_Scanner/", "volume/proj_512.volume");
	//sprintf(path, "%s%s", "C:/temp/", "volume/proj_512.volume");
	//sprintf(path, "%svolume/proj_%d.volume", para.str_main_folder.c_str(), para.nProj);
	volum.Load(path);
	//volum.FlipZ();
	//volum.FlipY();
	//ShowVolume(volum);
	//volum.GeneratorShape4();

	//double kernel[] = {0.2, 0.2, 0.2, 0.2, 0.2};
	//std::vector<double> vecKernel(kernel, kernel + sizeof(kernel)/sizeof(kernel[0]));
	//volum.FilterX(vecKernel);
	//volum.FilterY(vecKernel);
	//volum.FilterZ(vecKernel);

	//Rect32i roi(4, 2, para.nu-4*2, para.nv-2*2);

	//proj.SetRoi(roi);
	//CTBackProjection<float>(proj, para, volum);

	CTForwardProjection(para, volum, proj_2th);
	//CTForwardProjection(proj, proj_mask, para, volum, proj_2th);
	
	char szPath[1024];
	//sprintf(szPath, "/media/peng/DATADRIVE2/CtData/shepp-logan_w%d_h%d_c%04d/", para.nu, para.nv, para.nProj);
	sprintf(szPath, "/home/chen/projects/data/phantom3d/shepp-logan_w%d_h%d_c%04d/", para.nu, para.nv, para.nProj);
	proj_2th.SaveToFolder(szPath);
	printf("Done\n");
}
void Test3dReconstruction()
{
	DISPLAY_FUNCTION;
	//char szFolder[] = "D:/CT_Scanner/ct-rec-3d/Low960_768/";	
	//char szFolder[] = "c:/temp/Low960_768/";
	//char szFolder[] = "D:/CT_Scanner/High1920_1536/";	
	//char szFolder[] = "c:/temp/proj-img/";
	//char szFolder[] = "D:/CT_Scanner/proj-img/";
	char szPath[256]; 
	//////////////////////////////////////////////////
	Parameter para;
	para.str_proj_img_folder = "/home/chen/projects/data/phantom3d/shepp-logan_w608_h616_c0512/";
	/*	para.nx = para.ny = para.nz = 256;
		para.dx =  para.dy =  para.dz = 0.157000*2;

		para.nu = 256; //rows
		para.nv = 256; //colums
		//du = 0.2;  dv = 0.2;
		para.du = 0.7;  para.dv = 0.7;
		para.DSO = 468;   //source to object

		para.nProj = 180;     //512;
		para.vec_deg.resize(para.nProj);
		for (int i=0; i<para.nProj; i ++) para.vec_deg[i] = i *para.scan_angle/double(para.nProj);
		para.init();
		para.str_proj_img_folder = "C:/temp/3dCT-501/make_data/proj/";
		*/
	para.Display();
	ProjectionImageT<float> proj, proj_mask;
	Volume volum(para.nx, para.ny, para.nz);
	proj.LoadFolder<float>(para.str_proj_img_folder.c_str(), para.nu, para.nv, para.nProj, para.step);
	proj.ShiftToNoneNegative();

	proj_mask.LoadFolder<float>(para.str_mask_folder.c_str(), para.nu, para.nv, para.nProj, para.step);


	Rect32i roi(1, 1, para.nu - 1 * 2, para.nv - 1 * 2);

	proj.SetRoi(roi);
	CTBackProjection<float>(proj, para, volum);
	//CTBackProjection<float>(proj, proj_mask, para, volum);

	sprintf(szPath, "%s%s/proj_%d.volume", para.str_working_folder.c_str(), "volume", para.nProj);
	volum.Save(szPath);
}

void Iterative3dReconstructionRatio(int index = 0)
{
	DISPLAY_FUNCTION;
	try {
		index = 2;
		char szPath[1024] = "";	
		bool bRtn = false;
		const float kernel[] = { 0.05, 0.9, 0.05 };
		const std::vector<float> vecKernel(kernel, kernel + sizeof(kernel) / sizeof(kernel[0]));
		Parameter para;	
		para.str_filter = "none";
		para.nx = para.ny = para.nz = 512 / 2;
		para.dx =  para.dy =  para.dz = 0.157000 * 0.6 * 2;

		para.nu = 512;  //rows
		para.nv = 512;  //colums
		para.du = 0.2; para.dv = 0.2;

		para.nProj = 180;      //512;
		para.init();
		para.str_filter = "none";
		para.str_proj_img_folder = "C:/temp/3dCT-504/make_data/proj/";
		para.Display();
		const char* pWorkFolder = para.str_working_folder.c_str();


		ProjectionImage proj_ori, proj_2th, proj_ratio;
		Volume volume(para.nx, para.ny, para.nz);
		Volume Nor_volume(para.nx, para.ny, para.nz);
		Volume Ratio_volume(para.nx, para.ny, para.nz);

		{
			//////for test 
			//sprintf(szPath, "%svolume/Nor-%d.volume", pWorkFolder, para.nProj);
			//bRtn = Nor_volume.Load(szPath);
			//if (!bRtn){
			//	ProjectionImage proj_ones;
			//	VERIFY_TRUE(proj_ones.MallocBuffer(para.nu, para.nv, para.nProj, para.step));
			//	proj_ones.Ones();
			//	//proj_ones.SetRoi(roi);
			//	VERIFY_TRUE(CTBackProjection(proj_ones, para, Nor_volume)); 
			//	VERIFY_TRUE(Nor_volume.Save(szPath));
			//}else{
			//	printf("using saved Nor-volume\n");
			//}
			//assert(0);
		}

		const Rect32i roi(2, 2, para.nu - 2 * 2, para.nv - 2 * 2);
		bRtn = proj_ori.LoadFolder<ProjectionImage::DataType>(para.str_proj_img_folder.c_str(), para.nu, para.nv, para.nProj, para.step);
		proj_ori.ShiftToNoneNegative();

		VERIFY_TRUE(bRtn);
		VERIFY_TRUE(proj_2th.MallocBuffer(para.nu, para.nv, para.nProj, para.step).IsValid());
		VERIFY_TRUE(proj_ratio.MallocBuffer(para.nu, para.nv, para.nProj, para.step).IsValid());

		{
			//	//for test
			//	volume.Ones();
			//	VERIFY_TRUE(CTForwardProjection(proj_ori, para, volume, proj_2th));
		}

		int nIncrement = 0;
		int nStartIndex = 0;
		int nStatus = 0;
		for (nStartIndex = index - 1; nStartIndex >= 0; nStartIndex--) {
			for (int i = 0; i < 2; i++) {
				if (nStartIndex - i < 0) {
					nStartIndex = 0;
					nStatus     = 1;
					break;
				}
				sprintf(szPath, "%svolume/rec-%d-iter-%03d.volume", pWorkFolder, para.nProj, nStartIndex - i);
				bRtn = volume.Load(szPath);
				if (bRtn) {
					nIncrement = 0; 
					if (i == 0) {
						nStartIndex++;
						nStatus = 1;
					}
					else {
						nStatus = 2;
					}
					break;
				}
			}
			if (nStatus == 1) break;
			else if (nStatus == 0) continue;

			sprintf(szPath, "%s%03d/forward_proj/", pWorkFolder, nStartIndex);
			bRtn = proj_2th.LoadFolder<ProjectionImage::DataType>(szPath, para.nu, para.nv, para.nProj, para.step);
			if (bRtn) nIncrement = MAX(1, nIncrement); 
			//proj_2th.Display(szPath);

			sprintf(szPath, "%s%03d/forward_proj_ratio/", pWorkFolder, nStartIndex);
			bRtn = proj_ratio.LoadFolder<ProjectionImage::DataType>(szPath, para.nu, para.nv, para.nProj, para.step);
			if (bRtn) nIncrement = MAX(2, nIncrement); 
			//proj_ratio.Display(szPath);

			sprintf(szPath, "%svolume/rec-%d-iter-%03d-BackProj.volume", pWorkFolder, para.nProj, nStartIndex);
			bRtn = Ratio_volume.Load(szPath);
			if (bRtn) nIncrement = MAX(3, nIncrement);  
		
			break;
		}
		nStartIndex = MAX(nStartIndex, 0);

		proj_ori.SetRoi(roi);
		proj_2th.SetRoi(roi);
		proj_ratio.SetRoi(roi);

		printf("nIncrement = %d, nStartIndex = %d\n", nIncrement, nStartIndex);
		{
			if (nStartIndex == 0)
				volume.Ones();
			
			sprintf(szPath, "%svolume/Nor-%d.volume", pWorkFolder, para.nProj);
			bRtn = Nor_volume.Load(szPath);
			if (!bRtn) {
				ProjectionImage proj_ones;
				VERIFY_TRUE(proj_ones.MallocBuffer(para.nu, para.nv, para.nProj, para.step).IsValid());
				proj_ones.Ones();
				VERIFY_TRUE(para.str_filter == "none");
				//proj_ones /= (para.nProj/para.step);
				proj_ones.SetRoi(roi);
				VERIFY_TRUE(CTBackProjection(proj_ones, para, Nor_volume)); 
				VERIFY_TRUE(Nor_volume.Save(szPath));
			}
			else {
				printf("using saved Nor-volume\n");
			}

		}
		{
			for (int iter = MAX(0, nStartIndex); iter < 80; iter++) {
				printf("iterative reconstruction(%03d)----------------------\n", iter);
				if (nIncrement == 0) {
					VERIFY_TRUE(CTForwardProjection(proj_ori, para, volume, proj_2th));

					sprintf(szPath, "%s%03d/forward_proj/", pWorkFolder, iter);
					VERIFY_TRUE(proj_2th.SaveToFolder(szPath));
				}
				if (nIncrement == 1) nIncrement = 0;

				if (nIncrement == 0) {
					proj_ratio = proj_ori;
#if 1
					proj_ratio /= proj_2th;
#else
					proj_ratio -= proj_2th;
#endif

					sprintf(szPath, "%s%03d/forward_proj_ratio/", pWorkFolder, iter);
					proj_ratio.Display(szPath, &roi);
					VERIFY_TRUE(proj_ratio.SaveToFolder(szPath));
				}
				if (nIncrement == 2) nIncrement = 0;

				if (nIncrement == 0) {
					VERIFY_TRUE(para.str_filter == "none");
					//proj_ratio /= para.nProj/para.step;
					VERIFY_TRUE(CTBackProjection(proj_ratio, para, Ratio_volume));
					sprintf(szPath, "%svolume/rec-%d-iter-%03d-BackProj.volume", pWorkFolder, para.nProj, iter);
					VERIFY_TRUE(Ratio_volume.Save(szPath));
				}
				if (nIncrement == 3) nIncrement = 0;

				if (nIncrement == 0) {
					Ratio_volume /= Nor_volume;
					volume *= Ratio_volume;
					sprintf(szPath, "%svolume/rec-%d-iter-%03d.volume", pWorkFolder, para.nProj, iter);
					VERIFY_TRUE(volume.Save(szPath));
				}
				if (nIncrement == 4) nIncrement = 0;
			}
		}

	}
	catch (std::string& str) {
		std::cout << "Iterative3dReconstruction" + str + " error!" << std::endl;
	}
	catch (...) {
		std::cout << "Iterative3dReconstruction error!" << std::endl;
	}
}

void SART_3dReconstruction(int index = 0)
{
	DISPLAY_FUNCTION;
	try {
		index = 1;
		const int ITERATIVE_COUNT = 32;
		const float Lambda = 1.f / 3.f;
		char szPath[1024] = "";	
		bool bRtn = false;
		Parameter para;	
		para.str_filter = "none";
		para.Display();
		const char* pWorkFolder = para.str_working_folder.c_str();

		ProjectionImage proj_ori, proj_2th;
		Volume volume(para.nx, para.ny, para.nz);
		volume.Zeros();
		const Rect32i roi(2, 2, para.nu - 2 * 2, para.nv - 2 * 2);
		bRtn = proj_ori.LoadFolder<ProjectionImage::DataType>(para.str_proj_img_folder.c_str(), para.nu, para.nv, para.nProj, para.step);
		proj_ori.ShiftToNoneNegative();

		proj_2th.MallocBuffer(para.nu, para.nv, para.nProj, para.step);
		//proj_ori.Display("proj_ori");

		std::vector< unsigned int > projOrder(para.nProj / para.step);
		for (unsigned int i = 0; i < para.nProj; i++) projOrder[i] = i;
		std::random_shuffle(projOrder.begin(), projOrder.end());

		Vector3<float> szMM(para.dx*para.nx, para.dy*para.ny, para.dz*para.nz);
		const float scale = Lambda / szMM.Length();

		//volume.GeneratorShape6();
		for(int nIterIndex = 0 ; nIterIndex < ITERATIVE_COUNT ; nIterIndex++) {
			for (int nProjIndex = 0, idx = 0; nProjIndex < para.nProj; nProjIndex += para.step, idx++) {
				int _cur = (idx >= projOrder.size()) ? idx - projOrder.size() : idx;
				const int cur_proj_idx = projOrder[_cur];
				if (nIterIndex + nProjIndex > 0) {
					
				}
				if (cur_proj_idx == 54) {
					printf("cur_proj_idx = %d\n", cur_proj_idx);
				}
				{
					//forward projection
					//back projection
					//sub
					//multiple
					std::cout << "Iterative_Index = " << nIterIndex << ", idx = " << idx << std::endl;
					VERIFY_TRUE(CTForwardProjection(proj_ori, para, volume, proj_2th, cur_proj_idx));
					FrameT<ProjectionImage::DataType> frame_2th, frame_ori;
					proj_2th.GetFrame(frame_2th, cur_proj_idx);
					proj_ori.GetFrame(frame_ori, cur_proj_idx);
					//if (cur_proj_idx == 54)
					{
						//	WriteToFile<float, float>("C:/RtkData/054.raw", frame_2th.buffer, frame_2th.width*frame_2th.height);
					}
					//frame_2th.display("frame_2th");
					//frame_ori.display("frame_ori");
					//WaitKey(0);
					frame_2th -= frame_ori;
					//frame_2th.Sacle(-1);
					frame_2th.Sacle(scale * -1);
					VERIFY_TRUE(CTBackProjection(proj_2th, para, volume, cur_proj_idx));
				}
			}
			sprintf(szPath, "%s/proj-%d_ite_%d.volume", pWorkFolder, para.nProj, nIterIndex);
			volume.Save(szPath);
		}
		sprintf(szPath, "%s/proj-%d.volume", pWorkFolder, para.nProj);
		volume.Save(szPath);
	}
	catch (std::string& str) {
		std::cout << "Iterative3dReconstruction" + str + " error!" << std::endl;
	}
	catch (...) {
		std::cout << "Iterative3dReconstruction error!" << std::endl;
	}
}

void TestSART()
{
	typedef float DataType;
	ProjectionImageT<DataType> proj_ori;
	std::string path("C:/RtkData/proj_256x256x180_f32.raw");
	int width(256), height(256), count(180);
	VERIFY_TRUE(proj_ori.LoadRawData<float>(path, width, height, count));
	proj_ori.Display("proj_256x256x256_f32.raw");
}

void TestVolum()
{
	DISPLAY_FUNCTION;
	int i;
	Parameter para;
	char path[1024] = "C:/temp/3dCT-502/proj-512.volume";
	//char path[] = "C:/temp/volume/proj_360.volume";
	//char path[] = "C:/temp/volume/proj_512.volume";
	//sprintf(path, "%svolume/proj_%d.volume", para.str_main_folder.c_str(), para.nProj);
	//sprintf(path, "%svolume/proj_%d.volume", para.str_working_folder.c_str(), para.nProj);
	Volume vol;
	vol.Load(path);
	vol.ClearOutofRadiusData(vol.nx / 2.0 - 1);
	//Volume::DataType* pData = vol.GetDataZ(255);
	//SaveImageRaw16U("D:/CT_Scanner/slice.raw", pData, vol.nx, vol.ny, 0xfff, 0);
	//SaveImage16U("d:/slice.tif", pData, vol.nx, vol.ny, 0xfff, 0);

	//
	//for (i=0; i<512; i++){
	//	pData = vol.GetDataZ(i);
	//	char sztmp[1024];
	//	sprintf(sztmp, "D:/CT_Scanner/raw_slice/%03d.raw", i);
	//	SaveImageRaw16U(sztmp, pData, vol.nx, vol.ny, 0xffff, 0);
	//}	

	//vol *= -1;
	//vol.exp();
	//vol *= -1;
	double kernel[] = { 1.0 / 16, 4.0 / 16, 6.0 / 16, 4.0 / 16, 1.0 / 16 };
	std::vector<double> vecKernel(kernel, kernel + sizeof(kernel) / sizeof(kernel[0]));
	for (i = 0; ;) {
		if (i < 0) i = 511;
		if (i > 511) i = 0;
		i = MAX(0, MIN(i, 511));
		vol.SliceZ(i);
		vol.SliceY(i);
		vol.SliceX(i);	
		//vol.SaveRawZ("c:/temp/z.raw", i);
		printf("%d", i);
		printf("                                                 \r");
		int key = WaitKey();
		if (key == '+') i++;
		else if (key == '-') i--;
		else if (key == '\r') break;
		else if (key == 'x') vol.FilterX(vecKernel);
		else if (key == 'y') vol.FilterY(vecKernel);
		else if (key == 'z') vol.FilterZ(vecKernel);
		else if (key == '1') vol.FlipX();
		else if (key == '2') vol.FlipY();
		else if (key == '3') vol.FlipZ();
		else if (key == 's') {
			strcat(path, ".tmp");
			vol.Save(path);
		}
		else ;
	}
}


void TestIntersectBox() {
	Ray r;
	r.o = Float3(0, 0, 4);
	r.d = Float3(0, 0, -1);
	Float3 boxmin(-1, -1, -1);
	Float3 boxmax(1, 1, 1);
	float tnear = 0;
	float tfar = 0;

	IntersectBox(r, boxmin, boxmax, &tnear, &tfar);
}

void TestForwardBackwardProjection()
{
	DISPLAY_FUNCTION;
	bool bRtn;
	Parameter para;

	para.nx = para.ny = para.nz = 512;
	para.dx =  para.dy =  para.dz = 0.157000 * 0.6;

	para.nu = 512;  //rows
	para.nv = 512;  //colums
	//du = 0.2;  dv = 0.2;
	para.du = 0.2; para.dv = 0.2;

	para.nProj = 180;      //512;
	para.init();

	para.Display();
	char szPath[1024];
	const char* pWorkFolder = para.str_working_folder.c_str();
	
	ProjectionImage proj_one, proj_norm;
	proj_one.MallocBuffer(para.nu, para.nv, para.nProj, para.step);
	proj_norm.MallocBuffer(para.nu, para.nv, para.nProj, para.step);
	proj_one.Ones();
	proj_norm.Zero();

	//Volume volume_one(512, 512, 512);
	Volume volume_one(para.nx, para.ny, para.nz);
	Volume Norm_volume(para.nx, para.ny, para.nz);

	//volume_one.Ones();
	//volume_one.ClearOutofRadiusData(para.nx/4);
	//volume_one.GeneratorShape1();
	volume_one.GeneratorShape4();
	//volume_one.Resize(para.nx, para.ny, para.nz);
	volume_one.SliceZ(0);
	//WaitKey();

	VERIFY_TRUE(CTForwardProjection(proj_one, para, volume_one, proj_norm));
	sprintf(szPath, "%smake_data/proj/", pWorkFolder, para.nProj);
	//Add_Shift(proj_norm.Buffer(), 1024, proj_norm.Buffer(), proj_norm.width*proj_norm.height*proj_norm.GetCount());
	//Multiple_Scale(proj_norm.Buffer(), 256, proj_norm.Buffer(), proj_norm.width*proj_norm.height*proj_norm.GetCount());
	//Add_Shift(proj_norm.Buffer(), 1024, proj_norm.Buffer(), proj_norm.width*proj_norm.height*proj_norm.GetCount());
	proj_norm.SaveToFolder(szPath);
	
	VERIFY_TRUE(CTBackProjection(proj_norm, para, Norm_volume)); 
	sprintf(szPath, "%smake_data/make_proj-%d.volume", pWorkFolder, para.nProj);

	CheckImageQuality(volume_one, Norm_volume, -1024, 1024);
	Norm_volume.Save(szPath);

}

template<typename T> inline
ImageT<T> Difference(const ImageT<T>& src)
{
	ImageT<T> dst(src.width, src.height);
	dst.Clear();
	int radius = 1;
	for (int i = 0; i < src.height; i++) {
		for (int j = radius; j < src.width - 1; j++) {
			dst(j, i) = (src(j - 1, i) - src(j + 1, i)) * 0.5;
		}
	}
	return dst;
}

namespace tst
{
	typedef __int64 int64;
	struct SubVolume {
		SubVolume()
			: buffer(NULL)
			, rank(0)
			, size(0) {
		}
		virtual~SubVolume() {
			if (buffer) delete[]buffer; buffer = NULL;
		}
		BoxT<int64> subTop, subBottom;
		int nx, ny, nz;
		float* buffer;
		int64 rank, size;
	};

	struct DivVolume : public SubVolume {
		DivVolume(int64 _nx, int64 _ny, int64 _nz, int64 _rank, int64 _size) {
			nx = _nx; ny = _ny; nz = _nz;
			rank = _rank; 
			size = _size;
		
			double len = std::cbrt(double(_nx*_ny*_nz) / _size);
			int64 subNz = int64(len + 64 - 1 + 0.5) / 64 * 64;

			subNz = std::exp2(std::ceil(std::log2(subNz))) + 0.5; 
			VERIFY_TRUE(_nz % subNz == 0);
		
			int64 subNx = subNz;		
			int64 subNy = double(_nx*_ny*_nz) / (_size*subNx*subNz) + 0.5;	
			printf("Nx,Ny,Nz=%d,%d,%d, cnt=%d\n", subNx, subNy, subNz, _size);
		
			int64 sx = (_nx + subNx - 1) / subNx; 
			int64 sy = (_ny + subNy - 1) / subNy;
			int64 sz = (_nz + subNz - 1) / subNz;
			VERIFY_TRUE(sx*sy*sz <= _size);
		
			int z = _rank / (sx*sy);
			int y = (_rank - z*sx*sy) / sx;
			int x = _rank - z*sx*sy - y*sx;
		
			subTop = BoxT<int64>(x*subNx, y*subNy, z*subNz / 2, subNx, subNy, subNz / 2);
			subBottom = subTop;
			subBottom.z = _nz - 1 - subTop.z - subBottom.depth;
		
			buffer = new float[subTop.width*subTop.height*subTop.depth * 2];
			printf("malloc buffer : %f M\n", subTop.width*subTop.height*subTop.depth * 2.0 / 1024 / 1024);
		}	
	};

	void TestDivVolume()
	{
		int64 _nx = 2048;
		int64 _ny = 2048;
		int64 _nz = 2048;
		int64 _rank = 0;
		int64 _size = 64 * 2;
		DivVolume DV(_nx, _ny, _nz, _rank, _size);
	}
};

int main(int argc, char** argv)
{
	DISPLAY_FUNCTION;

	std::cout << "process_count = " << omp_get_num_procs() << std::endl;
	omp_set_num_threads(omp_get_num_procs() * 2);

	int index = 22;
	if (argc > 1) index = atoi(argv[1]);
	std::cout << "index = " << index << std::endl;
	switch (index) {
	case 0 :
		break;
	case 6:
		Test3dReconstruction();
		break;
	case 7:
		Test2dProjection();
		break;
	case 12:
		Iterative3dReconstructionRatio();
		break;
	case 21:
		TestForwardBackwardProjection();
		break;
	case 22:
		tst::TestDivVolume();
	default:
		break;
	}

	return 0;
}


