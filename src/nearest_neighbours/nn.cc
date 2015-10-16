/*
 * nn.cc
 *
 *  Created on: Oct 16, 2015
 *      Author: alexminnaar
 */

#include <iostream>
#include <ctime>
#include <cuda.h>
#include <cuda_runtime.h>				// Stops underlining of __global__
#include <device_launch_parameters.h>

using namespace std;

void FindClosestCPU(float3* points, int* indices, int count) {
// Base case, if there's 1 point don't do anything
if(count <= 1) return;
 // Loop through every point
for(int curPoint = 0; curPoint < count; curPoint++) {
	// This variable is nearest so far, set it to float.max
	float distToClosest = 3.40282e38f;
	// See how far it is from every other point
	for(int i = 0; i < count; i++) {
		// Don't check distance to itself
		if(i == curPoint) continue;
		float dist = ((points[curPoint].x - points[i].x) *
			(points[curPoint].x - points[i].x) +
			(points[curPoint].y - points[i].y) *
			(points[curPoint].y - points[i].y) +
			(points[curPoint].z - points[i].z) *
			(points[curPoint].z - points[i].z));
		if(dist < distToClosest) {
			distToClosest = dist;
			indices[curPoint] = i;
			}
		}
	}
}

int main()
{
// Number of points
const int count = 10000;

// Arrays of points
int *indexOfClosest = new int[count];
float3 *points = new float3[count];

// Create a list of random points
for(int i = 0; i < count; i++)
	{
	points[i].x = (float)((rand()%10000) - 5000);
	points[i].y = (float)((rand()%10000) - 5000);
	points[i].z = (float)((rand()%10000) - 5000);
	}

// This variable is used to keep track of the fastest time so far
long fastest = 1000000;

// Run the algorithm 20 times
for(int q = 0; q < 20; q++)
	{
	long startTime = clock();

	// Run the algorithm
	FindClosestCPU(points, indexOfClosest, count);

	long finishTime = clock();

	cout<<"Run "<<q<<" took "<<(finishTime - startTime)<<" millis"<<endl;

	// If that run was faster update the fastest time so far
	if((finishTime - startTime) < fastest)
		fastest = (finishTime - startTime);
	}

// Print out the fastest time
cout<<"Fastest time: "<<fastest<<endl;

// Print the final results to screen
cout<<"Final results:"<<endl;
for(int i = 0; i < 10; i++)
	cout<<i<<"."<<indexOfClosest[i]<<endl;

// Deallocate ram
delete[] indexOfClosest;
delete[] points;

return 0;
}
