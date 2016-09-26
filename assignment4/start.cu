#include <stdlib.h>
#include <stdio.h>
#include <cuda_runtime.h>
#include "utils.h"

#define VERBOSE 1
#define WINDOW_LENGTH 25
#define AXIS 3
#define OUTPUT_FILE "output.csv"

int main( int argc, char* argv[] )
{
	/* Local declarations */
	int *xcoord_loc, *ycoord_loc, *zcoord_loc;
	int* mag, *intense, *means, *max, *min;
	float* sd;
	int numOfLines, numOfReads;
	size_t coord_size, window_size;
		
/* Device declarations */
	int *tempx, *tempy, *tempz, *x_dev, *y_dev, *z_dev;
	int *mag_dev, *in_dev, *means_dev, *max_dev, *min_dev;
	float* sd_dev;
	cudaError_t err = cudaSuccess;

	/* Read in the data from CSV file */
	if( ( readData( &xcoord_loc, &ycoord_loc, &zcoord_loc, &numOfLines ) )!= 0 ){
		fprintf( stderr, "Reading of csv file failed!" );
		return 1;
	}
	
	/* Initializes some variables based on the data */
	numOfReads = numOfLines - (WINDOW_LENGTH - 1);
	coord_size = numOfLines * sizeof( int );
	window_size = numOfReads * WINDOW_LENGTH * sizeof( int );

	/* Allocate local result arrays */
	allocateLocal( &mag, &intense, &means, &sd, &max, &min, numOfReads );
	
	/* Allocates all device memory */
	allocate_dev( &tempx, window_size );
	allocate_dev( &tempy, window_size );
	allocate_dev( &tempz, window_size );
	allocate_dev( &x_dev, coord_size );
	allocate_dev( &y_dev, coord_size );
	allocate_dev( &z_dev, coord_size );
	allocate_dev( &mag_dev, numOfReads * sizeof(int) );
	allocate_dev( &in_dev, numOfReads * sizeof(int) );
	allocate_dev( &means_dev, numOfReads * AXIS * sizeof(int) );
	allocate_dev( &max_dev, numOfReads * AXIS * sizeof(int) );
	allocate_dev( &min_dev, numOfReads * AXIS * sizeof(int) );
	allocate_devf( &sd_dev, numOfReads * AXIS * sizeof(float) );

    /* Copy the original data coordinates to the device */
	err = cudaMemcpy( x_dev, xcoord_loc, coord_size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy xcoords from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    
	err = cudaMemcpy( y_dev, ycoord_loc, coord_size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy ycoords from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    
	err = cudaMemcpy( z_dev, zcoord_loc, coord_size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy zcoords from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

	/* Run massively parrallel CUDA function */
    int threadsPerBlock = 256;
    int blocksPerGrid =( numOfLines + threadsPerBlock - 1) / threadsPerBlock;
    
	cudaMagic<<<blocksPerGrid, threadsPerBlock>>>( mag_dev, in_dev, means_dev, sd_dev, max_dev, min_dev, tempx, tempy, tempz, x_dev, y_dev, z_dev, numOfLines ,numOfReads );
	err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to launch kernel (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    /* Copy the result arrays back to the host */
    err = cudaMemcpy( mag, mag_dev, numOfReads * sizeof(int), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy MAGNATUDE from device to host (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    
    err = cudaMemcpy( intense, in_dev, numOfReads * sizeof(int), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy INTENSITY from device to host (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    
    err = cudaMemcpy( means, means_dev, numOfReads * AXIS * sizeof(int), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy MEANS from device to host (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    
    err = cudaMemcpy( sd, sd_dev, numOfReads * AXIS * sizeof(float), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy STANDARD DEVIATION from device to host (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    
    err = cudaMemcpy( max, max_dev, numOfReads * AXIS * sizeof(int), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy MAX from device to host (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    
    err = cudaMemcpy( min, min_dev, numOfReads * AXIS * sizeof(int), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy MIN from device to host (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
	
	/* Writes out the final csv file */
	writeCSV( OUTPUT_FILE, &mag, &intense, &means, &sd, &max, &min, numOfReads);
	
	/* Free dynamic memory */
	free( xcoord_loc );
	free( ycoord_loc );
	free( zcoord_loc );
	free( mag );
	free( intense );
	free( means );
	free( sd );
	free( max );
	free( min );
	
	/* Free CUDA memory */
    cudaFree(x_dev);
	cudaFree(y_dev);
	cudaFree(z_dev);
	cudaFree(tempx);
	cudaFree(tempy);
	cudaFree(tempz);
    cudaFree( mag_dev );
    cudaFree( in_dev );
    cudaFree( means_dev );
    cudaFree( sd_dev );
    cudaFree( max_dev );
    cudaFree( min_dev );
}
