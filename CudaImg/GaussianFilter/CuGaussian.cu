/*================================================================
 * Display Images with CPU
 *----------------------------------------------------------------
 * Licence isn't exists.
 *
 * vmg.c
 *
 * Copyright (c) 2012 NULL
 *
 *================================================================*/

#include "Gaussian.h"
#include <cuda.h>
#include <cuda_runtime.h>


__global__ static void KerGaussianX(UINT32 *lpDst, UINT32 *lpSrc, float *lpCoeff, int width, int height, int radius);

__global__ static void KerGaussianY(UINT32 *lpDst, UINT32 *lpSrc, float *lpCoeff, int width, int height, int radius);

//__global__ static void KerGaussian(UINT32 *lpDst, UINT32 *lpSrc, float *lpCoeff, int width, int height, int radius);

__global__ static void KerMakeCoeff(float *lpCoeff, int radius)
{
	int	i = (blockDim.x * blockIdx.x + threadIdx.x);

	// Gaussian
	lpCoeff[i]	= 2.0 * ::exp(-(float)(i * i) / (float)(2 * radius * radius));

	// Lorentzian
//	lpCoeff[i]	= (256.0 / (1 + i * i));
}

__global__ static void KerGaussianX(UINT32 *lpDst, UINT32 *lpSrc, float *lpCoeff, int width, int height, int radius)
{
	int	x	= blockIdx.x * blockDim.x + threadIdx.x;
	int	y	= blockIdx.y * blockDim.y + threadIdx.y;
	int idx	= width * y + x;
	int	ratio;
	UINT32	pixel;
	float	r, g, b;
	int tx;
	float	totalRatio = 0;
	
	//if(x < radius || x > width - radius)
	//{return;}
	
	r = g = b = 0;
	totalRatio	= 0;

	for(int k = -radius; k <= radius; k ++)
	{
		tx	= x + k;
		if(tx >= 0 && tx < width)
		{
			pixel	= lpSrc[idx + k];
			//pixel	= lpSrc[y * width + tx];
			ratio	= lpCoeff[(int)::abs(k)];

			r += (0xFF & (pixel >> 16)) * ratio;
			g += (0xFF & (pixel >> 8)) * ratio;
			b += (0xFF & (pixel)) * ratio;

			totalRatio += ratio;
		}
	}

	r /= totalRatio;
	g /= totalRatio;
	b /= totalRatio;
			
	lpDst[idx]	= ((UINT32)r << 16) | ((UINT32)g << 8) | ((UINT32)b);
}

__global__ static void KerGaussianY(UINT32 *lpDst, UINT32 *lpSrc, float *lpCoeff, int width, int height, int radius)
{
	int	x	= blockIdx.x * blockDim.x + threadIdx.x;
	int	y	= blockIdx.y * blockDim.y + threadIdx.y;
	int idx	= width * y + x;
	int	ratio;
	UINT32	pixel;
	float	r, g, b;
	int ty;
	float	totalRatio = 0;
	
	r = g = b = 0;
	totalRatio	= 0;
			
	for(int k = -radius; k <= radius; k ++)
	{
		ty	= y + k;
		if(ty >= 0 && ty < height)
		{
			pixel	= lpSrc[ty * width + x];
			ratio	= lpCoeff[::abs(k)];

			r += (0xFF & (pixel >> 16)) * ratio;
			g += (0xFF & (pixel >> 8)) * ratio;
			b += (0xFF & (pixel)) * ratio;

			totalRatio += ratio;
		}
	}
			
	r /= totalRatio;
	g /= totalRatio;
	b /= totalRatio;
			
	lpDst[idx]	= ((UINT32)r << 16) | ((UINT32)g << 8) | ((UINT32)b);
}
//#include <cstdio>
#define BLOCKSIZE	16

void Imgproc::DCuGaussian(UINT32 *d_lpDst, UINT32 *d_lpSrc, UINT32 *d_lpTmp, float *d_lpCoeff, int width, int height, int radius)
{

	dim3	dimThread(BLOCKSIZE, BLOCKSIZE);
	dim3	dimBlock((width + BLOCKSIZE - 1) / BLOCKSIZE, (height + BLOCKSIZE - 1) / BLOCKSIZE);
	
	::KerMakeCoeff<<<radius, 1>>>(d_lpCoeff, radius);
//printf("%d, %d\n", width, height);

	/* Hrizontal bluring */
	::KerGaussianX<<<dimBlock, dimThread>>>(d_lpTmp, d_lpSrc, d_lpCoeff, width, height, radius);
	/* Vertical bluring (Source array "d_lpSrc" is used as destination) */
	::KerGaussianY<<<dimBlock, dimThread>>>(d_lpDst, d_lpTmp, d_lpCoeff, width, height, radius);
}

void Imgproc::CuGaussian(UINT32 *lpDst, UINT32 *lpSrc, int width, int height, int radius)
{
	UINT32	*d_lpSrc, *d_lpDst;
	float	*d_lpCoeff;

	::cudaMalloc((void **)&d_lpCoeff, sizeof(float) * radius);
	::cudaMalloc((void **)&d_lpSrc, sizeof(UINT32) * width * height);
	::cudaMalloc((void **)&d_lpDst, sizeof(UINT32) * width * height);
	::cudaMemcpy(d_lpSrc, lpSrc, sizeof(UINT32) * width * height, cudaMemcpyHostToDevice);

	::KerMakeCoeff<<<radius, 1>>>(d_lpCoeff, radius);

	dim3	dimThread(BLOCKSIZE, BLOCKSIZE);
	dim3	dimBlock((width + BLOCKSIZE - 1) / BLOCKSIZE, (height + BLOCKSIZE - 1) / BLOCKSIZE);

	/* Hrizontal bluring */
	::KerGaussianX<<<dimBlock, dimThread>>>(d_lpDst, d_lpSrc, d_lpCoeff, width, height, radius);
	/* Vertical bluring (Source array "d_lpSrc" is used as destination) */
	::KerGaussianY<<<dimBlock, dimThread>>>(d_lpSrc, d_lpDst, d_lpCoeff, width, height, radius);

	::cudaMemcpy(lpDst, d_lpSrc, sizeof(UINT32) * width * height, cudaMemcpyDeviceToHost);

	::cudaFree(d_lpCoeff);
	::cudaFree(d_lpSrc);
	::cudaFree(d_lpDst);
}