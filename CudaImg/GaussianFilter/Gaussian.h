/*================================================================
 * Display Images with CPU
 *----------------------------------------------------------------
 * Licence isn't exists.
 *
 * vmg.h
 *
 * Copyright (c) 2012 NULL
 *
 *================================================================*/

#pragma once

#include "Typedefs.h"

namespace Imgproc
{
	void Gaussian(UINT32 *lpDst, UINT32 *lpSrc, int width, int height, int radius);
	void GaussianX(UINT32 *lpDst, UINT32 *lpSrc, int *lpCoeff, int width, int height, int radius);
	void GaussianY(UINT32 *lpDst, UINT32 *lpSrc, int *lpCoeff, int width, int height, int radius);
	void SimdGaussianX(UINT32 *lpDst, UINT32 *lpSrc, int *lpCoeff, int width, int height, int radius);
	__attribute__((naked)) void SimdGaussianY(UINT32 *lpDst, UINT32 *lpSrc, int *lpCoeff, int width, int height, int radius);
	void DCuGaussian(UINT32 *d_lpDst, UINT32 *d_lpSrc, UINT32 *d_lpTmp, float *d_lpCoeff, int width, int height, int radius);
	void CuGaussian(UINT32 *lpDst, UINT32 *lpSrc, int width, int height, int radius);
};
