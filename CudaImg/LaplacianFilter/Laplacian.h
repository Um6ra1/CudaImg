/*================================================================
 * Laplacian filter
 *----------------------------------------------------------------
 * No licence, public domain.
 *
 * 
 *
 * Um6ra1
 *
 *================================================================*/

#pragma once

#include "Typedefs.h"

namespace Imgproc
{
	void Laplacian1(UINT32 *lpDst, UINT32 *lpSrc, int width, int height, int amplitude);
	void Laplacian2(UINT32 *lpDst, UINT32 *lpSrc, int width, int height, int amplitude);
	void AsmLaplacian(UINT32 *lpDst, UINT32 *lpSrc, int width, int height, int amplitude);
	void CuLaplacian(UINT32 *lpDst, UINT32 *lpSrc, int width, int height, int amplitude);
};
