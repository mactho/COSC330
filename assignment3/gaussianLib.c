
#include "gaussianLib.h"

/******************************************************************************
* generateGaussianKernel
* Generates a Gaussian kernel for a blur operation.
*
* Inputs:
* 
* kernel - Pointer to the allcated memory for the kernel
* kernel_dim - Square dimension of the kernel
* sd - Standard deviation of the gaussian distribution
* kerel_max - highest value in the kernel.
* colour_max - Sum of all colour values from kernel. Used
*              for normalisation to maintain correct colours
*
* Returns: Void
*
* 
* Reference: https://en.wikipedia.org/wiki/Gaussian_blur
******************************************************************************/
void
generateGaussianKernel (float **kernel, int kernel_dim, float sd, int origin,
			float *kernel_max, float *colour_max)
{
  int i, j;
  /*Initialise Accumulators/max vals */
  *colour_max = 0;
  *kernel_max = 0;
  /* Loop through and calculate each items value */
  for (i = 0; i < kernel_dim; i++)
    {
      double x_dist = abs (origin - i);
      for (j = 0; j < kernel_dim; j++)
	{
	  /*Calculate each term for the gaussian @x,y */
	  double y_dist = abs (origin - j);
	  double sqrt_term = sqrt (2.0 * M_PI * pow (sd, 2.0));
	  double exp_term =
	    exp (-
		 ((pow (x_dist, 2.0) +
		   pow (y_dist, 2.0)) / (2.0 * pow (sd, 2.0))));

	  kernel[i][j] = (1.0 / (sqrt_term)) * exp_term;

	  /* Check if new kernel_max is found */
	  if (kernel[i][j] > *kernel_max)
	    {
	      *kernel_max = kernel[i][j];
	    }
	  /*Accumulate colour_max */
	  *colour_max = *colour_max + (kernel[i][j] * 255.0);
	}
    }

}

/******************************************************************************
* GroundColorMix
* Produces an RGB colour based upon an input val x between
* min and max. Min must be greater than 0 and Max must be
* less than 256.
*
* Inputs:
* color - Pointer to double array of 3 elements. Ouput colour
*         goes in here.
* x - The input value
* min - Start point of the colour gradient (min > 0)
* max - End point of the colour gradient (max < 360)
******************************************************************************/
void
GroundColorMix (double *color, double x, double min, double max)
{
  /*
   * Red = 0
   * Green = 1
   * Blue = 2
   */
  double posSlope = (max - min) / 60;
  double negSlope = (min - max) / 60;

  /*Start at red */
  if (x < 60)
    {
      color[0] = max;
      color[1] = posSlope * x + min;
      color[2] = min;
      return;
    }
  else if (x < 120)
    {
      color[0] = negSlope * x + 2.0 * max + min;
      color[1] = max;
      color[2] = min;
      return;
    }
  else if (x < 180)
    {
      color[0] = min;
      color[1] = max;
      color[2] = posSlope * x - 2.0 * max + min;
      return;
    }
  else if (x < 240)
    {
      color[0] = min;
      color[1] = negSlope * x + 4.0 * max + min;
      color[2] = max;
      return;
    }
  else if (x < 300)
    {
      color[0] = posSlope * x - 4.0 * max + min;
      color[1] = min;
      color[2] = max;
      return;
    }
  /*End at blue-violet */
  else
    {
      color[0] = max;
      color[1] = min;
      color[2] = negSlope * x + 6 * max;
      return;
    }
}

/******************************************************************************
* bitmapFromSquareMatrix
* Builds a bitmap image based upon an input square matrix using the qdbmp lib.
* 
* Input:
* mat - The matris (already allocated and populated.
* filename - Name of the output bitmap.
* mat_dim - square dimension of matrix
* mat_max - max value present within the matrix
* colour_min - starting colour in gradient.
* colour_max - end colour in gradient.
*
* Returns: Void.
*
******************************************************************************/
void
bitmapFromSquareMatrix (float **mat, const char *filename, int mat_dim,
			float mat_max, float colour_min, float colour_max)
{
  double current_colour[3];
  int i, j;
  BMP *mat_out;
  mat_out = BMP_Create (mat_dim, mat_dim, 24);
  for (i = 0; i < mat_dim; i++)
    {
      for (j = 0; j < mat_dim; j++)
	{

	  /*Normalize values to lie between */
	  double curVal = colour_max - (((mat[i][j]) / mat_max) * colour_max);
	  GroundColorMix (current_colour, curVal, colour_min, colour_max);
	  BMP_SetPixelRGB (mat_out, j, i, current_colour[0],
			   current_colour[1], current_colour[2]);
	}
    }

  BMP_WriteFile (mat_out, filename);
  BMP_Free (mat_out);

}

/******************************************************************************
* applyConvolution
* Applies a convolution based upon a supplied kernel to an input bitmap. 
* Produces a new bitmap for the result. The convolution is applied by
* multiplying each kernel value with its corresponding pixel value and
* setting the sum of these values to the origin pixel. The output pixel
* is normalised so that it lies in the range 0-256.
*
* The convolution produces an image of the same size as the original,
* however the edges of the new image will be transformed assuming pixels
* beyond the edge of the image have a value of 0. This results in darker
* softened edges around the outside of the image.
*
* TODO: Implement edge bluring to avoid darkening (assuming this is not
* a feature!)
*
* Inputs:
* kernel - The kernel that will be used for the convolution.
* kernel_dim - The dimensions of the kernel (assumes square).
* colour_max - sum of all colours vals accross the kernel.
* old_bmp - bitmap to apply the convolution to.
* new_bmp - bitmap that will store the new convoluted image.
******************************************************************************/
void
applyConvolution (float **kernel, int kernel_dim, float kernel_origin,
		  float colour_max, BMP * old_bmp, BMP * new_bmp)
{
  /*Declare all the bits we need */
  unsigned char r, g, b;
  int x, y;
  unsigned int width, height;
  int kernel_start_x, kernel_start_y, kernel_end_x, kernel_end_y;
  int img_start_x, img_start_y;
  int kernel_x, kernel_y, img_x, img_y;
  float r_val = 0;
  float g_val = 0;
  float b_val = 0;
  /* Get image's dimensions */
  width = BMP_GetWidth (old_bmp);
  height = BMP_GetHeight (old_bmp);
  /*Iterate through all the image's pixels */
  /*Center pixel is at kernel_origin,kernel_origin */
  for (x = 0; x < width; ++x)
    {
      for (y = 0; y < height; ++y)
	{
	  /*Reset the RGB pixel value */
	  r_val = 0;
	  g_val = 0;
	  g_val = 0;

	  /* The calculations find the starting point within the kernel for the
	     current pixel. If the current pixel is more than kernel_origin from the
	     edge, the whole kernel is used and the starting point will be 0,0.
	     Otherwise we need to work out how much of the kernel will be used.
	     Remeber that the current pixel (x,y) will need to line up with the
	     pixel (kernel_origin,kernel_origin) from the kernel. */
	  if (x >= (kernel_origin))
	    {
	      kernel_start_x = 0;
	      img_start_x = x - (kernel_origin);
	    }
	  else
	    {
	      kernel_start_x = kernel_origin - x;
	      img_start_x = 0;
	    }
	  if (y >= (kernel_origin))
	    {
	      kernel_start_y = 0;
	      img_start_y = y - (kernel_origin);
	    }
	  else
	    {
	      kernel_start_y = kernel_origin - y;
	      img_start_y = 0;
	    }
	  if (width - x < (kernel_origin))
	    {
	      kernel_end_x = kernel_origin + (width - x);

	    }
	  else
	    {
	      kernel_end_x = kernel_dim;

	    }
	  if (height - y < (kernel_origin))
	    {
	      kernel_end_y = kernel_origin + (height - y);

	    }
	  else
	    {
	      kernel_end_y = kernel_dim;

	    }
	  /*Iterate through the pixels to calculate the final value for pixel x,y */
	  for (kernel_x = kernel_start_x, img_x = img_start_x;
	       kernel_x < kernel_end_x; kernel_x++, img_x++)
	    {
	      for (kernel_y = kernel_start_y, img_y = img_start_y;
		   kernel_y < kernel_end_y; kernel_y++, img_y++)
		{

		  BMP_GetPixelRGB (old_bmp, img_x, img_y, &r, &g, &b);
		  r_val = r_val + (r * kernel[kernel_x][kernel_y]);
		  b_val = b_val + (b * kernel[kernel_x][kernel_y]);
		  g_val = g_val + (g * kernel[kernel_x][kernel_y]);
		}

	    }
	  /*Normalise the new value to preserve the colours correctly */
	  r_val = (r_val / colour_max) * 255;
	  g_val = (g_val / colour_max) * 255;
	  b_val = (b_val / colour_max) * 255;
	  /*Apply the pixel to the output image */
	  BMP_SetPixelRGB (new_bmp, x, y, round (r_val), round (g_val),
			   round (b_val));

	}
    }



}
