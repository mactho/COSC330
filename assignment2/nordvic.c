/* Each thread represents a vehicle and is indeed assigned a vehicle
 * struct of it's own.  The bridge rules are:
 * - Uni directional at a time (1 lane)
 * - There can be multiple cars on the bridge at once as long as they
 * are all travelling the same direction.
 * - There can only ever be 1 truck on the bridge at once regardless
 * of the direction of travel */

#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <unistd.h>
#define THREADS 10
#define TRUCKS 3
#define CROSSINGTIME 4
#define MAXWAIT 20
#define VERBOSE 0

const static char east[] = "east";
const static char west[] = "west";
const static char car[] = "car";
const static char truck[] = "truck";
int static unsigned number_of_cars;
const static char* bridge_direction;
pthread_mutex_t static bridge_lock;
pthread_mutex_t static count_lock;
pthread_cond_t static cond;

typedef struct{
	int id;
	const char* type;
	const char* direction;
} vehicle;

/* Initializes all the vehicles */
void create_vehicles( vehicle* const v )
{
	if( TRUCKS > THREADS ){
		fprintf( stderr, "TRUCKS must be less than THREADS\n" );
		exit( EXIT_FAILURE );
	}
	int unsigned counter, car_id = 0;
	for( counter = 0 ; counter < THREADS ; counter++ ){
		if( counter < TRUCKS ){
			v[counter].type = truck;
			v[counter].id = counter;
		} else {
			v[counter].type = car;
			v[counter].id = car_id;
			car_id++;
		}
		/* Makes half the vehicles travel in each direction */
		if( ( counter % 2 ) == 0 ){
			v[counter].direction = east;
		} else {
			v[counter].direction = west;
		}
	}
}

/* Entry point for car type vehicles' threads */
void* cars( void* const ptr )
{
	const vehicle v = *( (vehicle*)ptr );
	sleep( rand() % MAXWAIT );
	if( VERBOSE ){
		printf( "%s %d has arrived at the bridge travelling %s bridge direction is %s\n", v.type, v.id, v.direction, bridge_direction );
	}
	pthread_mutex_lock( &count_lock );
	
	/* Only if the bridge direction is wrong and there are cars on the
	 * bridge do we block.*/
	if( !( bridge_direction == v.direction ) && !( number_of_cars == 0) ){
		pthread_cond_wait( &cond, &count_lock );
	}	
	number_of_cars++;
	/* If we are the first car on the bridge, lock the bridge direction */
	if( number_of_cars == 1 ){
		pthread_mutex_lock( &bridge_lock );
		bridge_direction = v.direction;
	}
	pthread_mutex_unlock( &count_lock );
	printf( "%s %d going %s on the bridge\n", v.type, v.id, v.direction );
	sleep( CROSSINGTIME );
	printf( "%s %d going %s off the bridge\n", v.type, v.id, v.direction );

	pthread_mutex_lock( &count_lock );
	number_of_cars--;
	if( number_of_cars == 0 ){
		if( VERBOSE )
			printf( "%s %d switching bridge direction..\n", v.type, v.id );
		if( bridge_direction == east ){
			bridge_direction = west;
		} else {
			bridge_direction = east;
		}
		/* unlock bridge direction */
		pthread_cond_broadcast( &cond );
		pthread_mutex_unlock( &bridge_lock );
	}
	pthread_mutex_unlock( &count_lock );
	
	pthread_exit( NULL );
}

/* Entry point for truck type vehicles' threads */
void* trucks( void* const ptr )
{
	const vehicle v = *( (vehicle*)ptr );
	sleep( rand() % MAXWAIT );
	if( VERBOSE ){
		printf( "%s %d has arrived at the bridge travelling %s bridge"
	" direction is %s\n", v.type, v.id, v.direction, bridge_direction );
	}
	pthread_mutex_lock( &bridge_lock );
	/* Mutual exclusion on bridge */
	bridge_direction = v.direction;
	printf( "%s %d going %s on the bridge\n", v.type, v.id, v.direction );
	sleep( CROSSINGTIME );
	printf( "%s %d going %s off the bridge\n", v.type, v.id, v.direction );
	/* End of mutual exclusion on bridge */
	pthread_mutex_unlock( &bridge_lock );
	pthread_exit( NULL );
}

/* Main function.  Initializes mutexes, Allocates memory for vehicles,
 * creates appropriate threads and directs them to the appropriate
 * functions, cleans up dynamic memory allocation */
int main( int argc, char* argv[] )
{
	int unsigned i;
	pthread_t* threads;
	vehicle* vehicle_data;
	
	/* Seed the rand() function with time so we don't keep getting the
	 * same sequence for all the waits at the bridge */
	 srand( time( NULL ) );
	
	/* Initialize bridge direction */
	bridge_direction = east;
	
	/* Initialize mutexes and condition variables */
	if( ( pthread_mutex_init( &count_lock, NULL ) ) != 0 ){
		fprintf( stderr, "count_lock initialization failed!\n" );
	}
	if( ( pthread_mutex_init( &bridge_lock, NULL ) ) != 0 ){
		fprintf( stderr, "bridge_lock initialization failed!\n" );
	}
	if( ( pthread_cond_init( &cond, NULL ) ) != 0 ){
		fprintf( stderr, "condition variable initialization failed\n" );
	}
	
	/* Allocate contiguous memory for thread id's */
	threads = (pthread_t*)calloc( THREADS, sizeof( pthread_t ) );
	/* Allocate contiguous memory for vehicle data */
	vehicle_data = (vehicle*)calloc( THREADS, sizeof( vehicle ) );
	/* Error checking for calloc system calls */
	if( ( vehicle_data == NULL ) || ( threads == NULL ) ){
		fprintf( stderr, "calloc failed, sorry!\n" );
	}
	
	/* initialize vehicles */
	create_vehicles( vehicle_data );
	
	/* Create threads with error checking */
	for( i = 0 ; i < THREADS ; i++ ){
		if( vehicle_data[i].type == truck ){ // sends thread to truck
			if( ( pthread_create( &threads[i], NULL, trucks, &vehicle_data[i] ) ) != 0 ){
				fprintf( stderr, "Thread creation failed!\n" );
			}
		} else { // sends car types to cars
			if( ( pthread_create( &threads[i], NULL, cars, &vehicle_data[i] ) ) != 0 ){
				fprintf( stderr, "Thread creation failed!\n" );
			}
		}
		if( VERBOSE ){
			printf( "Main thread: thread %d created.\n", i );
		}
	}
	
	/* Join threads back up with error checking */
	for( i = 0 ; i < THREADS ; i++ ){
		if( ( pthread_join( threads[i], NULL ) ) != 0 ){
			fprintf( stderr, "Join failed!\n" );
		}
	}
	
	/* Clean up with error checking */
	if( ( pthread_cond_destroy( &cond ) ) != 0 ){
		printf( "Could not destroy condition variable!\n" );
	}
	if( ( pthread_mutex_destroy( &count_lock ) ) != 0 ){
		printf( "Cound not destroy count mutex!\n" );
	}
	if( ( pthread_mutex_destroy( &bridge_lock ) ) != 0 ){
		printf( "Could not destroy bridge mutex!\n" );
	}
	free( threads );
	free( vehicle_data );
	exit( EXIT_SUCCESS );
}
