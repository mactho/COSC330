/******************************************************************************

	The program reads an BMP image file and creates a new
	image with gaussian blur applied.

******************************************************************************/

#include "gaussianLib.h"	/*Our Bitmap operations library */

#define KERNEL_DIMENSION_SD 3
#define KERNEL_BITMAP_FILENAME "kernel.bmp"

/******************************************************************************
* main
*
* Demo program for applying a gaussian blur filter to a bitmap image. This 
* program uses qdbmp library for bitmap operations.
*
* Usage: gaussian_blur <input file> <output file> <standard deviation>
*
* Compile: make
*
******************************************************************************/
int
main (int argc, char *argv[])
{
  BMP *bmp;
  BMP *new_bmp;
  float colour_max, kernel_max;
  int i;
  int width, height;
  float sd, kernel_dim, origin;
  float **kernel;

  /* Check arguments - You should check types as weel! */
  if (argc != 4)
    {
      fprintf (stderr,
	       "Usage: %s <input file> <output file> <standard deviation>\n",
	       argv[0]);
      return 0;
    }

  /*Standard Deviation of the Gaussian */
  sd = atoi (argv[3]);
  /*The kernel dimensions are deterined by th sd. Pixels beyond 3 standard deviations have 
     practiaclly on impact on the value for the origin cell */
  kernel_dim = (2 * (KERNEL_DIMENSION_SD * sd)) + 1;
  /*The center cell of the kernel */
  origin = KERNEL_DIMENSION_SD * sd;
  /* Now Lets allocate our Gaussian Kernel - The dimensions of the kernel will be 2*3*sd by 2*3*sd) */
  kernel = (float **) malloc (kernel_dim * sizeof (float *));
  for (i = 0; i < kernel_dim; i++)
    {
      kernel[i] = (float *) malloc (kernel_dim * sizeof (float));
    }

  /* Lets generate or kernel based upon the specs */
  generateGaussianKernel (kernel, kernel_dim, sd, origin, &kernel_max,
			  &colour_max);

  /* Lets create an image for the kernel just for a demo */
  bitmapFromSquareMatrix (kernel, KERNEL_BITMAP_FILENAME, kernel_dim,
			  kernel_max, 0, 255);

  /* Read an image file */
  bmp = BMP_ReadFile (argv[1]);
  BMP_CHECK_ERROR (stdout, -1);

  /* Lets check the runtime performance of our program */
  clock_t begin = clock ();

  width = BMP_GetWidth (bmp);
  height = BMP_GetHeight (bmp);
  new_bmp = BMP_Create (width, height, 24);

  applyConvolution (kernel, kernel_dim, origin, colour_max, bmp, new_bmp);

  clock_t end = clock ();
  double time_spent = (double) (end - begin) / CLOCKS_PER_SEC;
  printf ("Time spent processing:  %f\n", time_spent);

  /* Save result */
  BMP_WriteFile (new_bmp, argv[2]);
  BMP_CHECK_ERROR (stdout, -2);

  /* Free all memory allocated for the image */
  BMP_Free (bmp);
  BMP_Free (new_bmp);

  /* Free the kernel memory */
  for (i = 0; i < kernel_dim; i++)
    {
      free (kernel[i]);
    }
  free (kernel);

  return 0;
}
