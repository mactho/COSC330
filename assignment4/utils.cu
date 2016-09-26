#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <cuda_runtime.h>
#define BUFFSIZE 4096
#define WINDOW_LENGTH 25
#define AXIS 3
#define Y_INDEX 1
#define Z_INDEX 2

/* Does all the processing on the CUDA device.  Was intentionally NOT broken up into multiple functions for performance reasons, however is pretty well commented */
__global__ void cudaMagic( int* mag, int* in, int* means, float* sd, int* max, int* min, int* x, int* y, int* z, int* xcoords, int* ycoords, int* zcoords, int numOfLines, int length )
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	int count, tempCount, avg = 0, sdx = 0, sdy = 0, sdz = 0;
	int sumx = 0, sumy = 0, sumz = 0, absumx = 0, absumy = 0, absumz = 0;
	int xmax = 0, ymax = 0, zmax =0, xmin = 0, ymin = 0, zmin = 0;

	/*makes a flat arrays of all windows */
	if( i >= WINDOW_LENGTH && i <= numOfLines ){
		for( count = i - WINDOW_LENGTH, tempCount = 0 ; count < i ; count++, tempCount++ ){
			x[(i - WINDOW_LENGTH) * WINDOW_LENGTH + tempCount] = xcoords[count];
			y[(i - WINDOW_LENGTH) * WINDOW_LENGTH + tempCount] = ycoords[count];
			z[(i - WINDOW_LENGTH) * WINDOW_LENGTH + tempCount] = zcoords[count];
		}
	}
	__syncthreads();
	if( i < length ){
		/* Initialize the max and min values to the first value */
		xmax = x[i*WINDOW_LENGTH];
		ymax = y[i*WINDOW_LENGTH];
		zmax = z[i*WINDOW_LENGTH];
		xmin = x[i*WINDOW_LENGTH];
		ymin = y[i*WINDOW_LENGTH];
		zmin = z[i*WINDOW_LENGTH];

		for( count = i ; count < i + WINDOW_LENGTH ; count++ ){
			/* Calculates the sum of the absolute values for the window */
			absumx += fabsf(x[i * WINDOW_LENGTH + count]);
			absumy += fabsf(y[i * WINDOW_LENGTH + count]);
			absumz += fabsf(z[i * WINDOW_LENGTH + count]);
			/* Calculates the sums for the window */
			sumx += x[i * WINDOW_LENGTH + count];
			sumy += y[i * WINDOW_LENGTH + count];
			sumz += z[i * WINDOW_LENGTH + count];
			/* Calculates the average of the entire window */
			avg += (x[i * WINDOW_LENGTH + count] + y[i * WINDOW_LENGTH + count] + z[i * WINDOW_LENGTH + count] );
			/* Obtains the max coordinates for the window */
			xmax = fmaxf( x[i * WINDOW_LENGTH + count], xmax );
			ymax = fmaxf( y[i * WINDOW_LENGTH + count], ymax );
			zmax = fmaxf( z[i * WINDOW_LENGTH + count], zmax );
			/* Obtains the min coordinates for the window */
			xmin = fminf( x[i * WINDOW_LENGTH + count], xmin );
			ymin = fminf( y[i * WINDOW_LENGTH + count], ymin );
			zmin = fminf( z[i * WINDOW_LENGTH + count], zmin );
		}
		__syncthreads();
		/* Extra loop to calculate standard deviation because it relies on results of sumx, sumy, and sumz */
		for( count = 0 ; count < WINDOW_LENGTH ; count++ ){
			sdx += powf( (x[i * WINDOW_LENGTH + count] - (sumx/WINDOW_LENGTH)), 2 );
			sdy += powf( (y[i * WINDOW_LENGTH + count] - (sumy/WINDOW_LENGTH)), 2 );
			sdz += powf( (z[i * WINDOW_LENGTH + count] - (sumz/WINDOW_LENGTH)), 2 );
		}
		
		/* Writes all the results to their appropriate arrays */
		mag[i] = (absumx + absumy + absumz) / WINDOW_LENGTH;
		in[i] = avg / WINDOW_LENGTH;
		means[i] = sumx / WINDOW_LENGTH;
		means[Y_INDEX * length + i] = sumy / WINDOW_LENGTH;
		means[Z_INDEX * length + i] = sumz / WINDOW_LENGTH;
		sd[i] = sqrtf( sdx );
		sd[Y_INDEX * length + i] = sqrtf( sdy );
		sd[Z_INDEX * length + i] = sqrtf( sdz );
		max[i] = xmax;
		max[Y_INDEX * length + i] = ymax;
		max[Z_INDEX * length + i] = zmax;
		min[i] = xmin;
		min[Y_INDEX * length + i] = ymin;
		min[Z_INDEX * length + i] = zmin;
		
	}
}

/* Allocates device arrays with error checking (INT) */
__host__ void allocate_dev( int** array, const int size )
{
	cudaError_t err = cudaSuccess;
	err = cudaMalloc((void **)array, size );
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate array on device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

/* Allocates device arrays with error checking (FLOAT */
__host__ void allocate_devf( float** array, const int size )
{
	cudaError_t err = cudaSuccess;
	err = cudaMalloc((void **)array, size );
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate array on device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}
 
 /* Allocates the local arrays to copy the result arrays from the device to */
__host__ int allocateLocal( int** mag, int** in, int** means, float** sd, int** max, int** min, const int length )
{
	/* Allocate local result array (mag) */
	(*mag) = (int*)malloc( length * sizeof( int ) );
	(*in) = (int*)malloc( length * sizeof( int ) );
	(*means) = (int*)malloc( length * AXIS * sizeof( int ) );
	(*sd) = (float*)malloc( length * AXIS * sizeof( float ) );
	(*max) = (int*)malloc( length * AXIS * sizeof( int ) );
	(*min) = (int*)malloc( length * AXIS * sizeof( int ) );
	if( *mag == NULL || *in == NULL || *means == NULL || *sd == NULL || 
	*max == NULL || *min == NULL ){
		fprintf( stderr, "Malloc of local array failed!\n" );
		exit( EXIT_FAILURE );
	}
	return 0;
}

/* Returns the number of lines there are in the file pointed to by the argument fp */
__host__ int getLineCount( FILE** const fp )
{
	int count = 0;
	char c;
	
	/* Get the number of lines in the file */
	for( c = getc( *fp ) ; c != EOF ; c = getc( *fp ) ){
		if( c == '\n' ){
			count++;
		}
	}
	rewind( *fp );
	return count;
}

/*populates the local arrays with the data */
__host__ int readData( int** x, int** y, int** z, int* numOfLines )
{	
	char* token;
	char line[BUFFSIZE];
	int count;
	const char del[2] = ",";
	FILE* input;
	
	/* Open file for reading */
	if( ( input = fopen( "sheep_imu_data.csv", "r" ) ) == NULL ){
		fprintf( stderr, "Failed to open file!" );
		return 1;
	}
	
	*numOfLines = getLineCount( &input );
	
	*x = (int*)malloc( (*numOfLines) * sizeof(int) );
	*y = (int*)malloc( (*numOfLines) * sizeof(int) );
	*z = (int*)malloc( (*numOfLines) * sizeof(int) );
	if( *x == NULL || *y == NULL || *z == NULL ){
		fprintf( stderr, "Malloc of local array failed!\n" );
		exit( EXIT_FAILURE );
	}
	
	count = 0;
	while( fgets( line, BUFFSIZE, input ) ){
		token = strtok( line, del );
		(*x)[count] = atoi( token );
		token = strtok( NULL, del );
		(*y)[count] = atoi( token );
		token = strtok( NULL, del );
		(*z)[count] = atoi( token );
		count++;
	}
	
	fclose( input );
	return 0;
}

/* Writes arrays to a csv file */
__host__ void writeCSV( const char* const filename, int** mag, int** intensity, int** means, float** sd, int** max, int** min,  const int length )
{
	FILE* fp;
	unsigned int count;
	
	if( ( fp = fopen( filename, "w+" ) ) == NULL ){
		fprintf( stderr, "Failed to open or create new file!" );
	}
	
	for( count = 0 ; count < length ; count++ ){
		fprintf( fp, "%d, %d, %d, %d, %d, %f, %f, %f, %d, %d, %d, %d, %d, %d\n", (*mag)[count], (*intensity)[count], (*means)[count], (*means)[Y_INDEX*length+count], (*means)[Z_INDEX*length+count], (*sd)[count], (*sd)[Y_INDEX*length+count], (*sd)[Z_INDEX*length+count], (*max)[count], (*max)[Y_INDEX*length+count], (*max)[Z_INDEX*length+count], (*min)[count], (*min)[Y_INDEX*length+count], (*min)[Z_INDEX*length+count] );
	}
	fclose( fp );
}
