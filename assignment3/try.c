#include <stdio.h>
#include <stdlib.h>
#include <sys/stat.h>
#include "mpi.h"
#include "gaussianLib.h"
#define KERNEL_DIMENSION_SD 3
#define KERNEL_BITMAP_FILENAME "kernel.bmp"
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
	UCHAR *redBuf,*greenBuf, *blueBuf, *redSegment, *greenSegment, *blueSegment;
	
	/* Declare Matrices */
	UCHAR **redMat, **greenMat, **blueMat, **smallRedMat, **smallGreenMat, **smallBlueMat;
		
	unsigned int heightseg;
	unsigned long segLength;	

	BMP* bmp;
	BMP* temp;
	
	int index;
	float colour_max, kernel_max;
	int x,y, row, col;
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
	
	/*Standard Deviation of the Gaussian */
	sd = atoi (argv[3]);
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
	
	generateGaussianKernel (kernel, kernel_dim, sd, origin, &kernel_max, &colour_max);

		for( row = 0; row < kernel_dim; row++ ){
			free( kernel[row] );
		}
		free( kernel );
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
		
		/* Create large matrix to hold data for the entire input BMP */
		redMat = createMatrix( height, width );
		greenMat = createMatrix( height, width );
		blueMat = createMatrix( height, width );

		printf( "lllllll: %d\n", redMat[345][517] );
		printf( "Number of processes: %d\n", nproc );
		
		for( x = 0 ; x < width ; x++ ){
			for( y = 0 ; y < height ; y++ ){
				BMP_GetPixelRGB( bmp, x, y, &redMat[y][x], &greenMat[y][x], &blueMat[y][x] );
			}
		}
		if( DEBUG == 1 )printf( "last in array: %d %d %d\n", redMat[345][517], height, width );
		BMP_Free( bmp );
		
		/* Create send buffers for SCATTER functions */
		redBuf = (UCHAR *) calloc( nproc, segLength * sizeof( UCHAR ) );
		greenBuf = (UCHAR *) calloc( nproc, segLength * sizeof( UCHAR ) );
		blueBuf = (UCHAR *) calloc( nproc, segLength * sizeof( UCHAR ) );	
		
		/* Flatten matrix into buffers for SCATTER */
		index = 0;
		for( row = 0 ; row < height ; row++ ){
			for( col = 0 ; col < width ; col++ ){
				redBuf[index] = redMat[row][col];
				greenBuf[index] = greenMat[row][col];
				blueBuf[index] = blueMat[row][col];
				index++;
			}
		}
	}

	/* ALL NODES */

	/* Broadcast some necessary info to unpack the messages */
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
	}

	/* Allocate memory for the receive buffers */
	redSegment = (UCHAR *) calloc( segLength, sizeof( UCHAR ) );
	greenSegment = (UCHAR *) calloc( segLength, sizeof( UCHAR ) );
	blueSegment = (UCHAR *) calloc( segLength, sizeof( UCHAR ) ); 
	
	/* SCATTER the data */
	if( MPI_Scatter( redBuf, segLength, MPI_UNSIGNED_CHAR, redSegment, segLength, MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD ) != MPI_SUCCESS ){
		fprintf( stderr,"Scattering of redBuf failed\n" );
	}

	if( MPI_Scatter( greenBuf, segLength, MPI_UNSIGNED_CHAR, greenSegment, segLength, MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD ) != MPI_SUCCESS ){
		fprintf( stderr,"Scattering of greenBuf failed\n" );
	}
	
	if( MPI_Scatter( blueBuf, segLength, MPI_UNSIGNED_CHAR, blueSegment, segLength, MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD ) != MPI_SUCCESS ){
		fprintf( stderr,"Scattering of blueBuf failed\n" );
	}
	
	/* Allocate memory for small matrix to hold image fragment data after reconstitution */
	smallRedMat = createMatrix( heightseg, width );
	smallGreenMat = createMatrix( heightseg, width );
	smallBlueMat = createMatrix( heightseg, width );

	/* Reconstitute data from flattened segments into small matrix for processing !!!!! problem!!*/
	index = 0;
	for( row = 0 ; row < heightseg ; row++ ){
		for( col = 0 ; col < width ; col++ ){
			smallRedMat[row][col] = redSegment[index];
			smallGreenMat[row][col] = greenSegment[index];
			smallBlueMat[row][col] = blueSegment[index];
			index++;
		}
	}
	if( DEBUG == 1 )printf( "last check: %d\n", smallBlueMat[152][517] );
	
	if( DEBUG == 1 ){
		printf( "node# %d Seg: %ld\n", me, segLength );
		printf( "node# %d Width: %d\n", me, width );
		printf( "node# %d heightseg: %ld\n", me, heightseg );
		printf( "node# %d last: %d\n", me, blueSegment[segLength-1] );
		printf( "last mat: %d\n", smallBlueMat[152][517] );
	}
	/* AS SOON AS I TRY TO DO MORE IT ALL COLLAPSES */

	/* Create fragment size BMP to hold each fragment */
	temp = BMP_Create( (UINT)width, (UINT)heightseg, 24 );
	
	/*BMP_Free( temp ); */

	
	/* Populate the BMP with the data from the fragment matrix 
	for( row = 0 ; row < heightseg ; row++ ){
		for( col = 0 ; col < width ; col++ ){
			BMP_SetPixelRGB( temp, col + 1, row + 1, smallRedMat[row][col], smallGreenMat[row][col], smallBlueMat[row][col] );
		}
	} */
	/*blur = BMP_Create( width, heightseg, 24 );*/
/*	if( DEBUG == 1 ){
		printf( "node# %d Seg: %ld\n", me, segLength );
		printf( "node# %d Width: %d\n", me, width );
		printf( "node# %d heightseg: %ld\n", me, heightseg );
		printf( "node# %d last: %d\n", me, blueSegment[segLength-1] );
	}		
		BMP_WriteFile( temp, "temp.bmp" );
		new_bmp = BMP_Create( width, heightseg, 24 );
		BMP_Free( temp );	
*/		
		/* Free the kernel memory */		

	/*	
		for( row = 0 ; row < heightseg ; row++ ){
			free( smallRedMat[row] );
			free( smallGreenMat[row] );
			free( smallBlueMat[row] );
		}
		free( smallRedMat );
		free( smallGreenMat );
		free( smallBlueMat );
		
		free( redSegment );
		free( greenSegment );
		free( blueSegment );*/
	/*if( me == 0 ){
		/*
		for( row = 0 ; row < height ; row++ ){
			free( redMat[row] );
			free( greenMat[row] );
			free( blueMat[row] );
		}
		free( redMat );
		free( greenMat );
		free( blueMat );
		
		free( redBuf );
		free( greenBuf );
		free( blueBuf );
		
		/*
		
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
	MPI_Finalize();
	return 0;
}
