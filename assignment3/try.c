#include <stdio.h>
#include <stdlib.h>
#include <sys/stat.h>
#include "mpi.h"
#include "gaussianLib.h"
#define KERNEL_DIMENSION_SD 3
#define COLOUR_DEPTH 24
#define DEBUG 0

/* Argument parsing... Unfinished */
int parse_args( int argc, char *argv[] )
{
	if( ( argc != 4 ) || ( atoi( argv[3] ) <= 0 ) ){
		fprintf( stderr, "Usage: %s <input file> <output file> <standard deviation>\n", argv[0] );
		return( -1 );
	}
	return( 0 );
}

int main( int argc, char** argv )
{
	int nproc, me, err;
	
	/* Declare buffers */
	UCHAR *redBuf,*greenBuf, *blueBuf, *redSeg, *greenSeg, *blueSeg;

	/* Declare BMP'S */
	BMP *bmp, *temp1, *temp1_blur;
	
	unsigned int heightseg;
	unsigned int segLength;	
	int index;
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
	
	sd = atoi( argv[3] );
	kernel_dim = (2 * (KERNEL_DIMENSION_SD * sd)) + 1;
	origin = KERNEL_DIMENSION_SD * sd;
	kernel = (float **) malloc (kernel_dim * sizeof (float *));
	for (row = 0; row < kernel_dim; row++)
    {
      kernel[row] = (float *) malloc (kernel_dim * sizeof (float));
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
		
		if( DEBUG == 1 )printf( "Number of processes: %d\n", nproc );
		
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
	} 
	
	/* ALL NODES */

	/* Broadcast some necessary info to unpack the messages */
	if( MPI_Bcast( &segLength, 1, MPI_UNSIGNED, 0, MPI_COMM_WORLD ) != MPI_SUCCESS ){
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
	redSeg = (UCHAR *) calloc( segLength, sizeof( UCHAR ) );
	greenSeg = (UCHAR *) calloc( segLength, sizeof( UCHAR ) );
	blueSeg = (UCHAR *) calloc( segLength, sizeof( UCHAR ) );

	/* SCATTER the data */	
	if( MPI_Scatter( redBuf, segLength, MPI_UNSIGNED_CHAR, redSeg, segLength, MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD ) != MPI_SUCCESS ){
		fprintf( stderr,"Scattering of redBuf failed\n" );
	}
	
	if( MPI_Scatter( greenBuf, segLength, MPI_UNSIGNED_CHAR, greenSeg, segLength, MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD ) != MPI_SUCCESS ){
		fprintf( stderr,"Scattering of greenBuf failed\n" );
	}
	
	if( MPI_Scatter( blueBuf, segLength, MPI_UNSIGNED_CHAR, blueSeg, segLength, MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD ) != MPI_SUCCESS ){
		fprintf( stderr,"Scattering of blueBuf failed\n" );
	}

	if( DEBUG == 1 ){
		printf( "node# %d Seg: %d\n", me, segLength );
		printf( "node# %d Width: %d\n", me, width );
		printf( "node# %d heightseg: %d\n", me, heightseg );
		printf( "node# %d last: %d\n", me, blueSeg[segLength-1] );
	}

	/* Reconstitute the linear fragments into BMPs */
	temp1 = BMP_Create( width, heightseg, COLOUR_DEPTH );
	temp1_blur = BMP_Create( width, heightseg, COLOUR_DEPTH );
	
	index = 0;
	for( row = 0 ; row < heightseg ; row++ ){
		for( col = 0 ; col < width ; col++ ){
			BMP_SetPixelRGB( temp1, col, row, redSeg[index], greenSeg[index], blueSeg[index] );
			index++;
		}
	}
	
	/* Create the kernel and apply the blur convolution */
	generateGaussianKernel (kernel, kernel_dim, sd, origin, &kernel_max, &colour_max);
	applyConvolution( kernel, kernel_dim, origin, colour_max, temp1, temp1_blur );
	BMP_Free( temp1 );
	for( row = 0; row < kernel_dim; row++ ){
		free( kernel[row] );
	}
	free( kernel );

	/* Put blurred pixels back into segments */
	index = 0;
	for( row = 0 ; row < heightseg ; row++ ){
		for( col = 0 ; col < width ; col++ ){
			BMP_GetPixelRGB( temp1_blur, col, row, &redSeg[index], &greenSeg[index], &blueSeg[index] ); 
			index++;
		}
	}
	
	BMP_Free( temp1_blur );
	
	if( MPI_Gather( redSeg, segLength, MPI_UNSIGNED_CHAR, redBuf, segLength, MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD) != MPI_SUCCESS){
      fprintf( stderr, "Gathering of redBuf failed\n" );
    }
    
	if( MPI_Gather( greenSeg, segLength, MPI_UNSIGNED_CHAR, greenBuf, segLength, MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD) != MPI_SUCCESS){
      fprintf( stderr, "Gathering of greenBuf failed\n" );
    }
    
	if( MPI_Gather( blueSeg, segLength, MPI_UNSIGNED_CHAR, blueBuf, segLength, MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD) != MPI_SUCCESS){
      fprintf( stderr, "Gathering of blueBuf failed\n" );
    }
	
	free( redSeg );
	free( greenSeg );
	free( blueSeg );
	
	if( me == 0 ){
		index = 0;
		for( row = 0 ; row < height ; row++ ){
			for( col = 0 ; col < width ; col++ ){
				BMP_SetPixelRGB( bmp, col, row, redBuf[index], greenBuf[index], blueBuf[index] );
				index++;
			}
		}
		
		BMP_WriteFile( bmp, argv[2] );
		BMP_Free( bmp );
		
		free( redBuf );
		free( greenBuf );
		free( blueBuf );	
	}
	
	MPI_Barrier( MPI_COMM_WORLD );
	MPI_Finalize();
	return 0;
}
