#include "dil.h"




	#include <stdio.h>
	#include <cuda.h>

	#include "simple_arrays.h"

	typedef struct {
		int16_t x;
		int16_t y;
	} CudaPair;

	__global__ void cudaCreateSEKernel(const int32_t SE_ROWS, const int32_t SE_COLS,
			const float* D_SE, CudaPair* d_set_se, int32_t* d_set_n) {
		const int32_t IDX_ROW = blockIdx.x * blockDim.x + threadIdx.x;
		const int32_t IDX_COL = blockIdx.y * blockDim.y + threadIdx.y;
		const int32_t IDX = IDX_ROW * SE_COLS + IDX_COL;

		if (IDX_ROW >= SE_ROWS || IDX_COL >= SE_COLS || D_SE[IDX] == 0) {
			return;
		}

		CudaPair p;
		p.x = IDX_ROW - SE_ROWS / 2;
		p.y = IDX_COL - SE_COLS / 2;
		d_set_se[atomicAdd(d_set_n, 1)] = p;
	}

	template <int SE_CACHE_SIZE>
	__global__ void cudaBinaryDilationKernel(const int32_t IMG_ROWS,
			const int32_t IMG_COLS, const float* D_IMG, const int32_t SE_ROWS,
			const int32_t SE_COLS, const CudaPair* D_SET_SE,
			const int32_t SET_SE_N, float* d_dil) {

		__shared__ CudaPair
		d_s_se[SE_CACHE_SIZE];

		const int32_t IDX_ROW = blockIdx.x * blockDim.x + threadIdx.x;
		const int32_t IDX_COL = blockIdx.y * blockDim.y + threadIdx.y;
		const int32_t IDX = IDX_ROW * IMG_COLS + IDX_COL;

		if (IDX_ROW >= IMG_ROWS || IDX_COL >= IMG_COLS) {
			return;
		}

		const float L_T = D_IMG[IDX];
		if (L_T > 0) {
			d_dil[IDX] = 255.0f;
		}

		int32_t se_n_done = 0;

		const int32_t TIDX = threadIdx.x * blockDim.x + threadIdx.y;

	#pragma unroll
		while (se_n_done < SET_SE_N) {
			const int32_t SE_END =
			(se_n_done + SE_CACHE_SIZE < SET_SE_N) ?
			se_n_done + SE_CACHE_SIZE : SET_SE_N;
			const int32_t SE_DIFF = SE_END - se_n_done;
			if (se_n_done + TIDX < SE_END) {
				d_s_se[TIDX] = D_SET_SE[se_n_done + TIDX];
			}
			__syncthreads();

			if (L_T > 0) {
				for (int32_t i = 0; i < SE_DIFF; i++) {
					const CudaPair P = d_s_se[i];
					const int32_t I_X_MAX = (P.x + IDX_ROW > 0) ? P.x + IDX_ROW : 0;
					const int32_t I_Y_MAX = (P.y + IDX_COL > 0) ? P.y + IDX_COL : 0;

					d_dil[((I_X_MAX < IMG_ROWS - 1) ? I_X_MAX : IMG_ROWS - 1)
					* IMG_COLS + ((I_Y_MAX < IMG_COLS - 1) ?
							I_Y_MAX : IMG_COLS - 1)] = 255.0f;
				}
			}
			se_n_done += SE_DIFF;
			__syncthreads();
		}
	}

	Image32F* cudaBinaryDilation(const Image32F& H_IMG, const Image32F& H_SE) {

		/*
		 * Debug:
		 * Criação dos eventos para verificar tempo de execução
		 */
		cudaEvent_t start, stop;
		cudaEventCreate(&start);
		cudaEventCreate(&stop);
		cudaEventRecord(start, 0);

		/*
		 * Alocar memória na GPU para Imagem e Elemento Estruturante;
		 *
		 */
		const size_t IMG_N_BYTES = H_IMG.size * sizeof(float);
		float* d_img;
		cudaMalloc((void**) &d_img, IMG_N_BYTES);

		float* d_dil;
		cudaMalloc((void**) &d_dil, IMG_N_BYTES);
		cudaMemset(d_dil, 0, IMG_N_BYTES);

		const size_t SE_N_BYTES = H_SE.size * sizeof(float);
		float* d_se;
		cudaMalloc((void**) &d_se, SE_N_BYTES);

	// Imagino o pior caso ;)
		const size_t SET_SE_N_BYTES = H_SE.size * sizeof(CudaPair);
		CudaPair* d_set_se;
		cudaMalloc((void**) &d_set_se, SET_SE_N_BYTES);

		int32_t* d_set_se_n;
		cudaMalloc((void**) &d_set_se_n, sizeof(int32_t));
		cudaMemset(d_set_se_n, 0, sizeof(int32_t));

		/*
		 * Envio das imagens para a GPU
		 */
		cudaMemcpy(d_img, H_IMG.raster, IMG_N_BYTES, cudaMemcpyHostToDevice);
		cudaMemcpy(d_se, H_SE.raster, SE_N_BYTES, cudaMemcpyHostToDevice);

		const uint8_t BLK_SIZE = 8;
		const dim3 SE_DIM_GRID(H_SE.dims[0] / BLK_SIZE + 1,
				H_SE.dims[1] / BLK_SIZE + 1);
		const dim3 DIM_BLOCK(BLK_SIZE, BLK_SIZE);

		/*
		 * Crio um conjunto do Elemento Estruturante...
		 */
		cudaCreateSEKernel<<< SE_DIM_GRID, DIM_BLOCK >>>(
				H_SE.dims[0], H_SE.dims[1],
				d_se, d_set_se, d_set_se_n);

		const dim3 DIM_GRID(H_IMG.dims[0] / BLK_SIZE + 1,
				H_IMG.dims[1] / BLK_SIZE + 1);

		/*
		 * Invocação do Kernel da dilatação...
		 */
		int32_t h_set_se_n = 0;
		cudaMemcpy(&h_set_se_n, d_set_se_n, sizeof(int32_t),
				cudaMemcpyDeviceToHost);
		cudaBinaryDilationKernel<64><<< DIM_GRID, DIM_BLOCK >>>(
				H_IMG.dims[0], H_IMG.dims[1], d_img,
				H_SE.dims[0], H_SE.dims[1], d_set_se, h_set_se_n,
				d_dil);

		/*
		 * Imagem resultando da GPU pro Host...
		 */
		float* temp_img = (float*) malloc(IMG_N_BYTES);
		cudaMemcpy(temp_img, d_dil, IMG_N_BYTES, cudaMemcpyDeviceToHost);

		/*
		 * Limpando a bagunça...
		 */
		cudaFree(d_img);
		cudaFree(d_se);
		cudaFree(d_dil);
		cudaFree(d_set_se_n);
		cudaFree(d_set_se);

		Image32F* g = new Image32F();
		g->nd = H_IMG.nd;
		g->dims = new int[2];
		g->dims[0] = H_IMG.dims[0];
		g->dims[1] = H_IMG.dims[1];
		g->size = g->dims[0]*g->dims[1];
		g->raster = (char*) temp_img;

		/*
		 * Mostrando o tempo de processamento...
		 */
		cudaEventRecord(stop, 0);
		cudaEventSynchronize(stop);
		float elapsed_time;
		cudaEventElapsedTime(&elapsed_time, start, stop);
		printf("\n\ntime: %f ms\n", elapsed_time);
		cudaEventDestroy(start);
		cudaEventDestroy(stop);

		return g;
	}
