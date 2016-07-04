#include "FFTImage.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <cufft.h>
#include <thrust/extrema.h>
#include <thrust/execution_policy.h>
#include <thrust/device_ptr.h>

#include <cstdio>

#define BLOCK_DIM_X	16
#define BLOCK_DIM_Y	32

__global__ static void KerScrambleArray(float *d_lpDstRe, float *d_lpDstIm, float *d_lpSrcRe, float *d_lpSrcIm, int log2n);
__global__ static void KerFFT1D(float *d_lpDstRe, float *d_lpDstIm, int log2n);
__global__ static void KerIFFT1D(float *d_lpDstRe, float *d_lpDstIm, int log2n);
__global__ static void KerFFT2D(float *d_lpDstRe, float *d_lpDstIm, int log2n);
__global__ static void KerIFFT2D(float *d_lpDstRe, float *d_lpDstIm, int log2n);

__global__ static void KerBitReversalArray(float *d_lpDstRe, float *d_lpDstIm, float *d_lpSrcRe, float *d_lpSrcIm, int log2n)
{
	register int i = blockDim.x * blockIdx.x + threadIdx.x;
//	int length = 1 << log2n;

	if(i < (1 << log2n))
//	for(int i = 0; i < length; i ++)
	{
		register int idx = 0;
		register int x = i;
		
		for(int j = 0; j < log2n; j ++)
		{
			idx = (idx << 1) | (x & 1);
			x >>= 1;
		}

		if(i >= idx)
		{
			d_lpDstRe[idx]	= d_lpSrcRe[i];
			d_lpDstIm[idx]	= d_lpSrcIm[i];
			
			d_lpDstRe[i]	= d_lpSrcRe[idx];
			d_lpDstIm[i]	= d_lpSrcIm[idx];
		}
	}
}

#define N	16

void	BitRevTest()
{
	int x, y;
	float srcRe[N], srcIm[N], dstRe[N], dstIm[N];
	float *d_srcRe, *d_srcIm, *d_dstRe, *d_dstIm;

	for(int i = 0; i < N; i ++)
	{
		srcRe[i] = i;
		srcIm[i] = i * 10;
	}
	 
	::cudaMalloc((void **)&d_srcRe, sizeof(float) * N * 4);
	d_srcIm	= &d_srcRe[N * 1];
	d_dstRe	= &d_srcRe[N * 2];
	d_dstIm	= &d_srcRe[N * 3];
	
	::cudaMemcpy(d_srcRe, srcRe, sizeof(float) * N, cudaMemcpyHostToDevice);
	::cudaMemcpy(d_srcIm, srcIm, sizeof(float) * N, cudaMemcpyHostToDevice);
	 
	dim3	dimBlock(512, 1, 1);
	dim3	dimGrid((N + dimBlock.x - 1) / dimBlock.x, 1, 1);
	 ::KerBitReversalArray<<<dimGrid, dimBlock>>>(d_dstRe, d_dstIm, d_srcRe, d_srcIm, (int)::log2((float)N));
	 
 	::cudaMemcpy(dstRe, d_dstRe, sizeof(float) * N, cudaMemcpyDeviceToHost);
 	::cudaMemcpy(dstIm, d_dstIm, sizeof(float) * N, cudaMemcpyDeviceToHost);
	
 	for(int i = 0; i < N; i ++)
 	{ printf("(%f, %f) (%d, %d)\n", dstRe[i], dstIm[i], 0, 0); }
	
	::cudaFree(d_srcRe);
}

__global__ static void KerFFTX(float *d_lpDstRe, float *d_lpDstIm, int log2x)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	register int k = blockIdx.x * blockDim.x + threadIdx.x;
	UINT	i, j;
	UINT	l1, l2;
	//int	log2n	= (int)(::log2(length));
	int	width	= 1 << log2x;
//	int	halfLength	= length >> 1;
//	float	wRe, wIm, uRe, uIm;
	float	z, n, w1, w2, u1, u2;
	
	////////////////////////////////
	// FFT-X
	////////////////////////////////
	//for(int y = 0; y < height; y ++)
	{
		// ---- Bit reversal - X
		
		// ---- FFT - X
		w1	= -1.0;
		w2	= 0.0;
		l2	= 1;
		for(k = 0; k < log2x; k ++)
		{
			l1	= l2;
			l2	<<= 1;
			u1	= 1.0;
			u2	= 0.0;
		
			for(i = 0; i < l1; i ++)
			{
				for(j = i; j < width; j += l2)
				{
					register int	idx	= width * y + j + l1;
					register int	jdx	= width * y + j;
					register double	 tmpRe	= u1 * d_lpDstRe[idx] - u2 * d_lpDstIm[idx];
					register double	 tmpIm	= u1 * d_lpDstIm[idx] + u2 * d_lpDstRe[idx];

					d_lpDstRe[idx]	= d_lpDstRe[jdx] - tmpRe;
					d_lpDstIm[idx]	= d_lpDstIm[jdx] - tmpIm;
					d_lpDstRe[jdx]	+= tmpRe;
					d_lpDstIm[jdx]	+= tmpIm;
				}

				// (u1 + i u2) * (w1 + i w2)
				z	= u1 * w1 - u2 * w2;
				u2	= u1 * w2 + u2 * w1;
				u1	= z;
			}

			// \sin(x) = \sqrt{ \frac{\cos(2x) - 1}{-2} }
			// \cos(x) = \sqrt{ \frac{\cos(2x) + 1}{2} }
			w2	= -::sqrt((1.0 - w1) / 2.0);
			w1	= ::sqrt((1.0 + w1) / 2.0);
		}
	}

	//n = (float)length;
	n = ::sqrt((float)width);
	//for(i = 0; i < length; i ++)
	{
		d_lpDstRe[i]	/= n;
		d_lpDstIm[i]	/= n;
	}
}

__global__ static void KerBitReversalMatrixRow(float *d_lpDstRe, float *d_lpDstIm, float *d_lpSrcRe, float *d_lpSrcIm, int width, int log2x)
{
	register int x = blockDim.x * blockIdx.x + threadIdx.x;
	register int y = blockDim.y * blockIdx.y + threadIdx.y;
//	int width = 1 << log2x;

	if(x < (1 << log2x))
//	for(int i = 0; i < length; i ++)
	{
		register int index	= 0;
		register int t	= x;
		
		for(int j = 0; j < log2x; j ++)
		{
			index = (index << 1) | (t & 1);
			t >>= 1;
		}

		if(x >= index)
		{
			register int idx	= width * y + x;
			register int jdx	= width * y + index;

			register double	 tmpRe	= d_lpDstRe[idx];
			register double	 tmpIm	= d_lpDstIm[idx];
			
			d_lpDstRe[idx]	= d_lpSrcRe[jdx];
			d_lpDstIm[idx]	= d_lpSrcIm[jdx];
			
			d_lpDstRe[jdx]	= tmpRe;
			d_lpDstIm[jdx]	= tmpIm;
		}
	}
}

__global__ static void KerBitReversalMatrixCol(float *d_lpDstRe, float *d_lpDstIm, float *d_lpSrcRe, float *d_lpSrcIm, int width, int log2y)
{
	register int x = blockDim.x * blockIdx.x + threadIdx.x;
	register int y = blockDim.y * blockIdx.y + threadIdx.y;
//	int height = 1 << log2y;

	if(y < (1 << log2y))
//	for(int i = 0; i < length; i ++)
	{
		register int index	= 0;
		register int t	= y;
		
		for(int j = 0; j < log2y; j ++)
		{
			index = (index << 1) | (t & 1);
			t >>= 1;
		}

		if(y >= index)
		{
			register int idx	= width * y + x;
			register int jdx	= width * index + x;
			
			register double	 tmpRe	= d_lpDstRe[idx];
			register double	 tmpIm	= d_lpDstIm[idx];
			
			d_lpDstRe[idx]	= d_lpSrcRe[jdx];
			d_lpDstIm[idx]	= d_lpSrcIm[jdx];
			
			d_lpDstRe[jdx]	= tmpRe;
			d_lpDstIm[jdx]	= tmpIm;
		}
	}
}

static void FFT2D(float *d_lpDstRe, float *d_lpDstIm, float *d_lpSrcRe, float *d_lpSrcIm, float *d_lpTmp, int width, int height)
{
	UINT	i, j, k;
	UINT	l1, l2;
	int	log2x	= (int)::log2((float)width);
	int	log2y	= (int)::log2((float)height);
	int	halfWidth	= width >> 1;
	int	halfHeight	= height >> 1;
	double	wRe, wIm, uRe, uIm;
	double	z, n, w1, w2, u1, u2;
	
	dim3	dimBlock1D	= dim3(BLOCK_DIM_X * BLOCK_DIM_Y, 1, 1);
	dim3	dimGrid1D	= dim3((width * height + dimBlock1D.x - 1) / dimBlock1D.x, 1, 1);
	dim3	dimBlock2D	= dim3(BLOCK_DIM_X, BLOCK_DIM_Y, 1);
	dim3	dimGrid2D	= dim3((width / 2 + dimBlock2D.x - 1) / dimBlock2D.x, (height / 2 + dimBlock2D.y - 1) / dimBlock2D.y, 1);
	
	::KerBitReversalMatrixRow<<<dimGrid2D, dimBlock2D>>>(d_lpDstRe, d_lpDstIm, d_lpSrcRe, d_lpSrcIm, width, (int)::log2((float)width));
	::KerBitReversalMatrixCol<<<dimGrid2D, dimBlock2D>>>(d_lpDstRe, d_lpDstIm, d_lpSrcRe, d_lpSrcIm, width, (int)::log2((float)height));

#if 0
	////////////////////////////////
	// FFT-X
	////////////////////////////////
	for(int y = 0; y < height; y ++)
	{
		// ---- Bit reversal - X
		
		// ---- FFT - X
		w1	= -1.0;
		w2	= 0.0;
		l2	= 1;
		for(k = 0; k < log2x; k ++)
		{
			l1	= l2;
			l2	<<= 1;
			u1	= 1.0;
			u2	= 0.0;
		
			for(i = 0; i < l1; i ++)
			{
				for(j = i; j < width; j += l2)
				{
					register int	idx	= width * y + j + l1;
					register int	jdx	= width * y + j;
					register double	 tmpRe	= u1 * lpDstRe[idx] - u2 * lpDstIm[idx];
					register double	 tmpIm	= u1 * lpDstIm[idx] + u2 * lpDstRe[idx];

					lpDstRe[idx]	= lpDstRe[jdx] - tmpRe;
					lpDstIm[idx]	= lpDstIm[jdx] - tmpIm;
					lpDstRe[jdx]	+= tmpRe;
					lpDstIm[jdx]	+= tmpIm;
				}

				// (u1 + i u2) * (w1 + i w2)
				z	= u1 * w1 - u2 * w2;
				u2	= u1 * w2 + u2 * w1;
				u1	= z;
			}

			// \sin(x) = \sqrt{ \frac{\cos(2x) - 1}{-2} }
			// \cos(x) = \sqrt{ \frac{\cos(2x) + 1}{2} }
			w2	= -::sqrt((1.0 - w1) / 2.0);
			w1	= ::sqrt((1.0 + w1) / 2.0);
		}
	}
	
	////////////////////////////////
	// FFT-Y
	////////////////////////////////
	for(int x = 0; x < width; x ++)
	{
		// ---- Bit reversal - Y

		// ---- FFT - Y
		w1	= -1.0;
		w2	= 0.0;
		l2	= 1;
		for(k = 0; k < log2y; k ++)
		{
			l1	= l2;
			l2	<<= 1;
			u1	= 1.0;
			u2	= 0.0;
		
			for(i = 0; i < l1; i ++)
			{
				for(j = i; j < height; j += l2)
				{
					register int	idx	= width * (j + l1) + x;
					register int	jdx	= width * j + x;
					register double	 tmpRe	= u1 * lpDstRe[idx] - u2 * lpDstIm[idx];
					register double	 tmpIm	= u1 * lpDstIm[idx] + u2 * lpDstRe[idx];

					lpDstRe[idx]	= lpDstRe[jdx] - tmpRe;
					lpDstIm[idx]	= lpDstIm[jdx] - tmpIm;
					lpDstRe[jdx]	+= tmpRe;
					lpDstIm[jdx]	+= tmpIm;
				}

				// (u1 + i u2) * (w1 + i w2)
				z	= u1 * w1 - u2 * w2;
				u2	= u1 * w2 + u2 * w1;
				u1	= z;
			}

			// \sin(x) = \sqrt{ \frac{\cos(2x) - 1}{-2} }
			// \cos(x) = \sqrt{ \frac{\cos(2x) + 1}{2} }
			w2	= -::sqrt((1.0 - w1) / 2.0);
			w1	= ::sqrt((1.0 + w1) / 2.0);
		}
	}

	// ---- Scaling
	//return;
	//n = (double)length;
	n = ::sqrt((double)width * height);
	for(i = 0; i < width * height; i ++)
	{
		lpDstRe[i]	/= n;
		lpDstIm[i]	/= n;
	}
#endif
}

__global__ void NormalizeDFT2D(cuComplex *d_lpDst, float scalar)
{
	register int idx = blockIdx.x * blockDim.x + threadIdx.x;

	d_lpDst[idx].x /= scalar;
	d_lpDst[idx].y /= scalar;
}

__global__ void ToComplex(cuComplex *d_lpDst, UINT32 *d_lpSrc, int shift)
{
	register int idx = blockIdx.x * blockDim.x + threadIdx.x;

	d_lpDst[idx].x = (float)((d_lpSrc[idx] >> shift) & 0xFF);
	d_lpDst[idx].y = 0;
}

__global__ void ToSpectrum(float *d_lpDst, cuComplex *d_lpSrc)
{
	register int idx = blockIdx.x * blockDim.x + threadIdx.x;
	register float amp;
	register float norm2	= d_lpSrc[idx].x * d_lpSrc[idx].x + d_lpSrc[idx].y * d_lpSrc[idx].y;
	//	norm2	= d_lpDstRe[i] * d_lpDstRe[i] + d_lpDstIm[i] * d_lpDstIm[i];
	//amp	= ::abs(::atan2(d_lpDstIm[i], d_lpDstRe[i]));

	if(norm2 >= 1.0) {	amp = ::log10(norm2) / 2.0; }
	else { amp = 0.0; }
	
	d_lpDst[idx] = amp;
}

__global__ void ToIntImageOverride(UINT32 *d_lpDst, float *d_lpSrc, int width, int height, float ampMax)
{
	register int x = blockIdx.x * blockDim.x + threadIdx.x;
	register int y = blockIdx.y * blockDim.y + threadIdx.y;

	register int idx = y * width + x;

	d_lpDst[idx + (height + 1) * width / 2]	= (UINT32)(d_lpSrc[idx] * 255.0 / ampMax);
	d_lpDst[idx + width / 2]				= (UINT32)(d_lpSrc[idx + height * width / 2] * 255.0 / ampMax);
	d_lpDst[idx]						= (UINT32)(d_lpSrc[idx + (height + 1) * width / 2] * 255.0 / ampMax);
	d_lpDst[idx + height * width / 2]		= (UINT32)(d_lpSrc[idx + width / 2] * 255.0 / ampMax);
}

__global__ void ToIntImage(UINT32 *d_lpDst, float *d_lpSrc, int width, int height, float ampMax, int shift)
{
	register int x = blockIdx.x * blockDim.x + threadIdx.x;
	register int y = blockIdx.y * blockDim.y + threadIdx.y;

	register int idx = y * width + x;

	d_lpDst[idx + (height + 1) * width / 2]	|= (UINT32)(d_lpSrc[idx] * 255.0 / ampMax) << shift;
	d_lpDst[idx + width / 2]				|= (UINT32)(d_lpSrc[idx + height * width / 2] * 255.0 / ampMax) << shift;
	d_lpDst[idx]						|= (UINT32)(d_lpSrc[idx + (height + 1) * width / 2] * 255.0 / ampMax) << shift;
	d_lpDst[idx + height * width / 2]		|= (UINT32)(d_lpSrc[idx + width / 2] * 255.0 / ampMax) << shift;
}

void Imgproc::DCuFFTImage(UINT32 *d_lpDst, UINT32 *d_lpSrc, float *d_lpTmp, int width, int height)
{
	int	idx;
	int	i, j, k;
//	int ftWidth	= 1 << (int)(::ceil(::log2((float)width)));
//	int ftHeight	= 1 << (int)(::ceil(::log2((float)height)));
	float	*d_lpSrcRe	= d_lpTmp;	// !!!! This is not compatible with "cuComplex"
	float	*d_lpSrcIm	= &d_lpTmp[width * height * 1];
	float	*d_lpDstRe	= &d_lpTmp[width * height * 2];
	float	*d_lpDstIm	= &d_lpTmp[width * height * 3];
	cuComplex	*d_lpTmpSrc	= (cuComplex *)d_lpTmp;
	cuComplex	*d_lpTmpDst	= (cuComplex *)&d_lpTmp[width * height * 2];
	dim3	dimBlock1D	= dim3(BLOCK_DIM_X * BLOCK_DIM_Y, 1, 1);
	dim3	dimGrid1D	= dim3((width * height + dimBlock1D.x - 1) / dimBlock1D.x, 1, 1);
	dim3	dimBlock2D	= dim3(BLOCK_DIM_X, BLOCK_DIM_Y, 1);
	dim3	dimGrid2D	= dim3((width / 2 + dimBlock2D.x - 1) / dimBlock2D.x, (height / 2 + dimBlock2D.y - 1) / dimBlock2D.y, 1);
	cufftHandle fftPlan;
	float	ampMax;

//	BitRevTest();
//	return;

	::cufftPlan2d(&fftPlan, width, height, CUFFT_C2C);
	//#define USE_CUFFT
	for(k = 0; k < 24; k += 8)
	{
#ifdef USE_CUFFT
		// ---- x => (x, 0)
		::ToComplex<<<dimGrid1D, dimBlock1D>>>(d_lpTmpSrc, d_lpSrc, k);

		// ---- Fourier Transform
		::cufftExecC2C(fftPlan, d_lpTmpSrc, d_lpTmpSrc, CUFFT_FORWARD);
#else
		FFT2D(d_lpDstRe, d_lpDstIm, d_lpSrcRe, d_lpSrcIm, d_lpTmp, width, height);
#endif
		// ---- (X, Y) => log(X^2 + Y^2) / 2
		//::NormalizeDFT2D<<<dimGrid, dimBlock>>>(d_lpTmpSrc, ::sqrt(width * height));
		::ToSpectrum<<<dimGrid1D, dimBlock1D>>>(d_lpDstRe, d_lpTmpSrc);
		
		// ---- Get max value from spectrum
		ampMax = thrust::reduce(thrust::device_ptr<float>(d_lpDstRe), thrust::device_ptr<float>(d_lpDstRe + (width * height)), -1, thrust::maximum<float>());;
//		printf("%f\n", ampMax);

		// ---- Spectrum to image
		if(k == 0)
		{
			::ToIntImageOverride<<<dimGrid2D, dimBlock2D>>>(d_lpDst, d_lpDstRe, width, height, ampMax);
		}
		else
		{
			::ToIntImage<<<dimGrid2D, dimBlock2D>>>(d_lpDst, d_lpDstRe, width, height, ampMax, k);
		}
	}

	::cufftDestroy(fftPlan);
}

