#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <algorithm>
#include <iostream>
#include <vector>
#include <string>
#include <time.h>
#include <chrono>
#include <thread>

using namespace std;
using namespace std::this_thread;
using namespace std::chrono_literals;
using std::chrono::system_clock;

// Helper function to generate random double between set intervals.
__device__ __host__ double doubleRand( double min , double max )
{
	double random = ( double ) rand() / RAND_MAX;
	return min + random * ( max - min );
}

class Racer
{
private:
	double		speed;
	double		pos;
	string		name;

public:
	__device__ __host__ Racer() : speed ( 0.0 ) , pos ( 0.0 ) , name( "" )
	{
		speed = doubleRand( 1.0 , 5.0 );
	}
	__device__ __host__ Racer( string name ) : speed( 0.0 ) , pos( 0.0 ) , name( name )
	{
		speed = doubleRand( 1.0 , 5.0 );
	}
	__device__ __host__ Racer( const Racer* other )
	{
		this->speed = other-> speed;
		this->pos = other->pos;
		this->name = other->name;
	}
	__device__ __host__ ~Racer() {}
	__device__ __host__ double getSpeed() const { return speed; }
	__device__ __host__ double getPos() const { return pos; }
	__device__ __host__ string getName() const { return name; }
	__device__ __host__ bool getWinnerState()
	{
		if( pos > 100.0 )
		{
			return true;
		}
		return false;
	}
	__device__ __host__ void calcNewPos()
	{
		pos += getSpeed();
	}
	__host__ void print()
	{
		cout << getName() << ": " << getPos() << "m\t" << "Speed: " << getSpeed() << "m/s" << endl;
	}

	bool operator < ( const Racer& other ) const
	{
		return this->getPos() < other.getPos();
	}
};

// Momentary printing
__host__ void raceStatus( Racer* list , int size )
{
	system( "CLS" );

	cout << "--- Stage Info ---" << endl;

	for( int i = 0; i < size; i ++ )
	{
		list[ i ].print();
	}
}

// Print the final results
__host__ void finalResults( Racer* list , int size )
{
	cout << "\n--- Final Results ---" << endl;

	vector< Racer > results;

	for( int j = 0; j < size; j++ )
	{
		results.push_back( list[ j ] );
	}

	std::sort( results.begin() , results.end() );

	for( int j = 0; j < size; j++ )
	{
		cout << j + 1;

		if( j == 10 || j == 11 || j == 12 )
		{
			cout << "th";
		}
		else
		{
			if( j % 10 < 3 )
			{
				if( j % 10 == 0 )
				{
					cout << "st";
				}
				else if( j % 10 == 1 )
				{
					cout << "nd";
				}
				else if( j % 10 == 2 )
				{
					cout << "rd";
				}
			}
			else
			{
				cout << "th";
			}
		}
		cout << " => " << results[ 99 - j ].getName() << "\t";

		if( (j + 1) % 4 == 0 )
		{
			cout << endl;
		}
	}

}

// GPU calculating new position of racers
__global__ void calcPos( Racer* list , int n )
{
	int i = threadIdx.x;

	list[ i ].calcNewPos();

	return;
}

// Helper function for using CUDA to calculate Racer positions.
cudaError_t calcWithCuda( Racer* list , int size )
{
	Racer *dev_list = 0;
	cudaError_t cudaStatus;

	// Choose which GPU to run on
	cudaStatus = cudaSetDevice( 0 );
	if( cudaStatus != cudaSuccess )
	{
		fprintf( stderr , "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?\n" );
		goto Error;
	}

	// Allocate GPU buffers for list of Racers
	cudaStatus = cudaMalloc( ( void** ) &dev_list , size * sizeof( Racer ) );
	if( cudaStatus != cudaSuccess )
	{
		fprintf( stderr , "cudaMalloc failed!\n" );
		goto Error;
	}

	// Copy input list from host to device
	cudaStatus = cudaMemcpy( dev_list , list , size * sizeof( Racer ) , cudaMemcpyHostToDevice );
	if( cudaStatus != cudaSuccess )
	{
		fprintf( stderr , "cudaMemcpy failed!\n" );
		goto Error;
	}

	// Launch a kernel on the GPU with one thread per Racer until we have a winner.
	while( true )
	{

		calcPos <<<1 , size >>> ( dev_list , size );

		// Check for any errors launching the kernel
		cudaStatus = cudaGetLastError();
		if( cudaStatus != cudaSuccess )
		{
			fprintf( stderr , "calcPos launch failed: %s\n" , cudaGetErrorString( cudaStatus ) );
			goto Error;
		}
	
		// cudaDeviceSynchronize waits for the kernel to finish, and returns
		// any errors encountered during the launch.
		cudaStatus = cudaDeviceSynchronize();
		if( cudaStatus != cudaSuccess )
		{
			fprintf( stderr , "cudaDeviceSynchronize returned error code %d after launching addKernel!\n" , cudaStatus );
			goto Error;
		}
	
		cudaStatus = cudaMemcpy ( list , dev_list , size * sizeof( Racer ) , cudaMemcpyDeviceToHost );
		if( cudaStatus != cudaSuccess )
		{
			fprintf( stderr , "cudaMemcpy failed!\n" );
			goto Error;
		}

		// Every time we return the racer list to the host, we must check if we have a winner.
		// Printing the race status at every iteration is entirely optional.
		raceStatus( list , 100 );

		for( int i = 0; i < size; i++ )
		{
			if( list[ i ].getWinnerState() )
			{
				// Invoke a false error which will leave the while loop.
				goto Error;
			}
		}


		sleep_for( 1s );
	}


	Error:
		cudaFree( dev_list );

	return cudaStatus;

}

int main()
{
	srand ( time( NULL ) );

	Racer host_list[ 100 ];
	for( int k = 0; k < 100; k++ )
	{
		// For simplicity, I am just giving numbers as their names.
		host_list[ k ] = new Racer( to_string( k ) );
	}

	cudaError_t cudaStatus = calcWithCuda( host_list , 100 );
	if( cudaStatus != cudaSuccess )
	{
		fprintf( stderr , "calcWithCuda failed!\n" );
		return -1;
	}

	finalResults( host_list , 100 );

	cudaStatus = cudaDeviceReset();
	if( cudaStatus != cudaSuccess )
	{
		fprintf( stderr , "cudaDeviceReset failed!\n" );
		return -1;
	}
	
	return 0;
}
