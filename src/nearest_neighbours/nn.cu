
/**
 * Copyright 1993-2012 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 */
#include <iostream>
#include <ctime>
#include <cuda.h>
#include <cuda_runtime.h>				// Stops underlining of __global__
#include <device_launch_parameters.h>	// Stops underlining of threadIdx etc.

using namespace std;


__global__ void FindClosestGPU(float3* points, int* indices, int count)
{
if(count <= 1) return;

int idx = threadIdx.x + blockIdx.x * blockDim.x;
if(idx < count)
	{
	float3 thisPoint = points[idx];
	float smallestSoFar = 3.40282e38f;

	for(int i = 0; i < count; i++)
		{
		if(i == idx) continue;

		float dist = (thisPoint.x - points[i].x)*(thisPoint.x - points[i].x);
		dist += (thisPoint.y - points[i].y)*(thisPoint.y - points[i].y);
		dist += (thisPoint.z - points[i].z)*(thisPoint.z - points[i].z);

		if(dist < smallestSoFar)
			{
			smallestSoFar = dist;
			indices[idx] = i;
			}
		}
	}
}



int main(int argc, char **argv)
{
	cout<<"running GPU implementation"<<endl;
	// Number of points
	const int count = 10000;

	// Arrays of points
	int *indexOfClosest = new int[count];
	float3 *points = new float3[count];
	float3* d_points;	 // GPU version
	int* d_indexOfClosest;

	// Create a list of random points
	for(int i = 0; i < count; i++)
		{
		points[i].x = (float)((rand()%10000) - 5000);
		points[i].y = (float)((rand()%10000) - 5000);
		points[i].z = (float)((rand()%10000) - 5000);
		}

	cudaMalloc(&d_points, sizeof(float3) * count);
	cudaMemcpy(d_points, points, sizeof(float3) * count, cudaMemcpyHostToDevice);
	cudaMalloc(&d_indexOfClosest, sizeof(int) * count);

	// This variable is used to keep track of the fastest time so far
	long fastest = 1000000;

	// Run the algorithm 20 times
	for(int q = 0; q < 20; q++)
		{
		long startTime = clock();

		// Run the algorithm
		//FindClosestCPU(points, indexOfClosest, count);

		FindClosestGPU<<<(count / 320)+1, 320>>>(d_points, d_indexOfClosest, count);
		cudaMemcpy(indexOfClosest, d_indexOfClosest, sizeof(int) * count, cudaMemcpyDeviceToHost);

		long finishTime = clock();

		cout<<q<<" "<<(finishTime - startTime)<<endl;

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
	cudaFree(d_points);
	cudaFree(d_indexOfClosest);

	cudaDeviceReset();

	return 0;
}
