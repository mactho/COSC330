__global__ void cudaMagic( int* mag, int* in, int* means, float* sd, int* max, int* min, int* x, int* y, int* z, int* xcoords, int* ycoords, int* zcoords, int numOfLines, int length );
__host__ void allocate_dev( int** array, int size );
__host__ void allocate_devf( float** array, int size );
__host__ int allocateLocal( int** mag, int** in, int** means, float** sd, int** max, int** min, int length );
__host__ int getLineCount( FILE** const fp );
__host__ int readData( int** x, int** y, int** z, int* numOfLines );
__host__ void writeCSV( const char* const filename, int** mag, int** intensity, int** means, float** sd, int** max, int** min, int length );
