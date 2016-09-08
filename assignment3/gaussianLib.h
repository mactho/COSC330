
#include "qdbmp.h"		/*Our Bitmap operations library */
#include <stdio.h>		/*For I/O */
#include <math.h>		/*For maths operations e.g pow,sqrt */
#include <stdlib.h>		/*For utils e.g. malloc */
#include <time.h>		/*For clock(...) operations */

void generateGaussianKernel (float **kernel, int kernel_dim, float sd,
			     int origin, float *kernel_max,
			     float *colour_max);
void GroundColorMix (double *color, double x, double min, double max);
void bitmapFromSquareMatrix (float **mat, const char *filename, int mat_dim,
			     float mat_max, float colour_min,
			     float colour_max);
void applyConvolution (float **kernel, int kernel_dim, float kernel_origin,
		       float colour_max, BMP * old_bmp, BMP * new_bmp);
