#include <stdio.h>
#include <stdlib.h>
#include <sys/stat.h>
#include "mpi.h"
#include "gaussianLib.h"
#define KERNEL_DIMENSION_SD 3
#define KERNEL_BITMAP_FILENAME "kernel.bmp"
#define COLOUR_DEPTH 24
#define DEBUG 1

/* Argument parsing... Unfinished */
int parse_args( int argc, char *argv[] )
{
	if( ( argc != 4 ) || ( atoi( argv[3] ) <= 0 ) ){
		fprintf( stderr, "Usage: %s <input file> <output file> <standard deviation>\n", argv[0] );
		return( -1 );
	}
	return( 0 );
}

/* Allocates memory for matrix of the desired dimensions */
UCHAR** createMatrix( int rows, int cols )
{
	UCHAR** mat;
	unsigned int counter;
	mat = (UCHAR**)calloc( rows, sizeof( UCHAR* ) );
	if( mat == NULL ){
		fprintf( stderr, "Calloc failed!\n" );
	}
	for( counter = 0 ; counter < rows ; counter++ ){
		mat[counter] = (UCHAR*)calloc( cols, sizeof( UCHAR ) );
		if( mat[counter] == NULL ){
			fprintf( stderr, "Calloc failed!\n" );
		}
	}
	return mat;
}

int main( int argc, char** argv )
{
	int nproc, me, err;
	
	/* Declare buffers */
	UCHAR *redBuf,*greenBuf, *blueBuf, *redSeg, *greenSeg, *blueSeg;
	UCHAR *redSeg1, *greenSeg1, *blueSeg1;	
	unsigned int heightseg;
	unsigned long segLength;	

	BMP* bmp;
	BMP* temp1;
	BMP* temp2;
	BMP* temp1_blur;
	BMP* temp2_blur;
	
	int index, overlap;
	float colour_max, kernel_max;
	int row, col;
	int width, height;
	float sd, kernel_dim, origin;
	float **kernel;
	
	/* Standard MPI Initialization */
	if( ( err = MPI_Init( &argc, &argv ) ) != MPI_SUCCESS ){
		fprintf( stderr, "MPI initialization failed!...Exiting" );
		return err;
	}	
	if( ( err = MPI_Comm_rank( MPI_COMM_WORLD, &me ) ) != MPI_SUCCESS ){
		fprintf( stderr, "Setting up ranks for processes failed!...Exiting" );
		return err;
	}
	if( ( err = MPI_Comm_size( MPI_COMM_WORLD, &nproc ) ) != MPI_SUCCESS ){
		fprintf( stderr, "Setting up Comm size failed!....Exiting" );
		return err;
	}
	
	/* setup the gaussian blur kernel and create it */
	printf( "sd is: %d\n", atoi( argv[3] ) );
	/*Standard Deviation of the Gaussian */
	sd = atoi( argv[3] );
	/*The kernel dimensions are deterined by th sd. Pixels beyond 3 standard deviations have 
	practiaclly on impact on the value for the origin cell */
	kernel_dim = (2 * (KERNEL_DIMENSION_SD * sd)) + 1;
	
	/*The center cell of the kernel */
	origin = KERNEL_DIMENSION_SD * sd;
	/* Now Lets allocate our Gaussian Kernel - The dimensions of the kernel will be 2*3*sd by 2*3*sd) */
	kernel = (float **) malloc (kernel_dim * sizeof (float *));
	for (row = 0; row < kernel_dim; row++){
		kernel[row] = (float *) malloc( kernel_dim * sizeof (float) );
	}
	
		
	/* ROOT ONLY !!! */
	if( me == 0 ){

		if( parse_args( argc, argv ) != 0 ){
			exit( EXIT_FAILURE );
		}
		
		bmp = BMP_ReadFile (argv[1]);
		BMP_CHECK_ERROR (stdout, -1);
		
		width = BMP_GetWidth( bmp );
		height = BMP_GetHeight( bmp );
		
		heightseg = height / nproc;
		segLength = heightseg * width;
		overlap = (int)( KERNEL_DIMENSION_SD * sd );
		
		printf( "Number of processes: %d\n", nproc );
		
		/* Create send buffers for SCATTER functions */
		redBuf = (UCHAR *)calloc( width * height, sizeof( UCHAR ) );
		greenBuf = (UCHAR *)calloc( width * height, sizeof( UCHAR ) );
		blueBuf = (UCHAR *)calloc( width * height,  sizeof( UCHAR ) );	
		
		/* Flatten image into buffers for SCATTER */
		index = 0;
		for( row = 0 ; row < height ; row++ ){
			for( col = 0 ; col < width ; col++ ){
				BMP_GetPixelRGB( bmp, col, row, &redBuf[index], &greenBuf[index], &blueBuf[index] ); 
				index++;
			}
		}
		BMP_Free( bmp );
		
		/* break the buffers into chucks, normally will be done by scatter */
		redSeg = (UCHAR *) calloc( segLength, sizeof( UCHAR ) );
		greenSeg = (UCHAR *) calloc( segLength, sizeof( UCHAR ) );
		blueSeg = (UCHAR *) calloc( segLength, sizeof( UCHAR ) );
		redSeg1 = (UCHAR *) calloc( segLength, sizeof( UCHAR ) );
		greenSeg1 = (UCHAR *) calloc( segLength, sizeof( UCHAR ) );
		blueSeg1 = (UCHAR *) calloc( segLength, sizeof( UCHAR ) );
		
		for( row = 0 ; row < height * width ; row++ ){
			if( row < segLength ){
				redSeg[row] = redBuf[row];
				greenSeg[row] = greenBuf[row];
				blueSeg[row] = blueBuf[row];
			} else {
				redSeg1[row -segLength] = redBuf[row];
				greenSeg1[row - segLength] = greenBuf[row];
				blueSeg1[row - segLength] = blueBuf[row];
			}
		}
		
		temp1 = BMP_Create( width, heightseg, COLOUR_DEPTH );
		temp2 = BMP_Create( width, heightseg, COLOUR_DEPTH );
		temp1_blur = BMP_Create( width, heightseg, COLOUR_DEPTH );
		temp2_blur = BMP_Create( width, heightseg, COLOUR_DEPTH );
		
		index = 0;
		for( row = 0 ; row < heightseg ; row++ ){
			for( col = 0 ; col < width ; col++ ){
				BMP_SetPixelRGB( temp1, col, row, redSeg[index], greenSeg[index], blueSeg[index] );
				BMP_SetPixelRGB( temp2, col, row, redSeg1[index], greenSeg1[index], blueSeg1[index] );
				index++;
			}
		}
	
		generateGaussianKernel (kernel, kernel_dim, sd, origin, &kernel_max, &colour_max);
	
		applyConvolution( kernel, kernel_dim, origin, colour_max, temp1, temp1_blur );
		applyConvolution( kernel, kernel_dim, origin, colour_max, temp2, temp2_blur );
		
		
		BMP_WriteFile( temp1_blur, "temp1.bmp" );
		BMP_WriteFile( temp2_blur, "temp2.bmp" );
		
		BMP_Free( temp1 );
		BMP_Free( temp2 );

		for( row = 0; row < kernel_dim; row++ ){
			free( kernel[row] );
		}
		free( kernel );
	}
	
	/* ALL NODES */

	/* Broadcast some necessary info to unpack the messages 
	if( MPI_Bcast( &segLength, 1, MPI_INT, 0, MPI_COMM_WORLD ) != MPI_SUCCESS ){
		fprintf( stderr,"Broadcast of seglength failed\n" );
	}
	if(	MPI_Bcast( &heightseg, 1, MPI_UNSIGNED, 0, MPI_COMM_WORLD ) != MPI_SUCCESS ){
		fprintf( stderr,"Broadcast of heightseg failed\n" );
	}
	if( MPI_Bcast( &width, 1, MPI_INT, 0, MPI_COMM_WORLD ) != MPI_SUCCESS ){
		fprintf( stderr,"Broadcast of width failed\n" );
	}
	if( MPI_Bcast( &height, 1, MPI_INT, 0, MPI_COMM_WORLD ) != MPI_SUCCESS ){
		fprintf( stderr,"Broadcast of height failed\n" ); 
	}*/

	/* Allocate memory for the receive buffers */

	
	/* SCATTER the data */
/*	
	if( MPI_Scatter( &redBuf, segLength, MPI_UNSIGNED_CHAR, &redSegment, segLength, MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD ) != MPI_SUCCESS ){
		fprintf( stderr,"Scattering of redBuf failed\n" );
	}

	if( MPI_Scatter( &greenBuf, segLength, MPI_UNSIGNED_CHAR, &greenSegment, segLength, MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD ) != MPI_SUCCESS ){
		fprintf( stderr,"Scattering of greenBuf failed\n" );
	}
	
	if( MPI_Scatter( &blueBuf, segLength, MPI_UNSIGNED_CHAR, &blueSegment, segLength, MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD ) != MPI_SUCCESS ){
		fprintf( stderr,"Scattering of blueBuf failed\n" );
	}
	
	if( DEBUG == 1 ){
		printf( "node# %d Seg: %ld\n", me, segLength );
		printf( "node# %d Width: %d\n", me, width );
		printf( "node# %d heightseg: %d\n", me, heightseg );
		printf( "node# %d last: %d\n", me, blueSegment[segLength-1] );
	}
*/	
	
	MPI_Finalize();
	return 0;
}
	
	/* Reconstitute data from flattened segments into small matrix for processing !!!!! problem!!*/
/*	
	temp = BMP_Create( width, height, COLOUR_DEPTH );
	
	index = 0;
	for( row = 0 ; row < heightseg ; row++ ){
		for( col = 0 ; col < width ; col++ ){
			BMP_SetPixelRGB( temp, col, row, redSegment[index], greenSegment[index], blueSegment[index] );
			index++;
		}
	}
	
	 AS SOON AS I TRY TO DO MORE IT ALL COLLAPSES 

	blur = BMP_Create( width, heightseg, 24 );
	if( DEBUG == 1 ){
		printf( "node# %d Seg: %ld\n", me, segLength );
		printf( "node# %d Width: %d\n", me, width );
		printf( "node# %d heightseg: %ld\n", me, heightseg );
		printf( "node# %d last: %d\n", me, blueSegment[segLength-1] );
	}		
		BMP_WriteFile( temp, "temp.bmp" );
		new_bmp = BMP_Create( width, heightseg, 24 );
		BMP_Free( temp );	
*/				

	/*	
		
		free( redSegment );
		free( greenSegment );
		free( blueSegment );*/
	/*if( me == 0 ){
		
		free( redBuf );
		free( greenBuf );
		free( blueBuf );
		
		
		
		redBuf = (UCHAR *) calloc( nproc, segLength * sizeof( UCHAR ) );
		blur = BMP_Create( width, heightseg, 24 );
		BMP_Free( blur );
		printf( "?????: %d, %ld\n", width, heightseg );


		new_bmp = BMP_Create( width, heightseg, 24 );
		BMP_Free( new_bmp );

		applyConvolution( kernel, kernel_dim, origin, colour_max, temp, new_bmp);
		
		BMP_WriteFile (new_bmp, "chunk.bmp");
		BMP_CHECK_ERROR (stdout, -2);
		
		 */
/*MPI_Barrier( MPI_COMM_WORLD );*/

	
	/* ROOT NODE */
		/*applyConvolution( kernel, kernel_dim, origin, colour_max, bmp, new_bmp);
		BMP_WriteFile (new_bmp, argv[2]);
		BMP_CHECK_ERROR (stdout, -2); */
		
		/*BMP_Free ( new_bmp );
		

	}*/

