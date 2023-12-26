/*
 * Based on CSC materials from:
 * 
 * https://github.com/csc-training/openacc/tree/master/exercises/heat
 *
 */
#include <algorithm>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define BLOCK_SIZE_X 16
#define BLOCK_SIZE_Y 16

/* Convert 2D index layout to unrolled 1D layout
 *
 * \param[in] i      Row index
 * \param[in] j      Column index
 * \param[in] width  The width of the area
 * 
 * \returns An index in the unrolled 1D array.
 */
int __host__ __device__ getIndex(const int i, const int j, const int width)
{
    return i*width + j;
}

__global__ void evolve_kernel(const double* Un, double* Unp1, const int nx, const int ny, const double dx2, const double dy2, const double aTimesDt)
{
    __shared__ double s_Un[(BLOCK_SIZE_X + 2)*(BLOCK_SIZE_Y + 2)];
    int i = threadIdx.x + blockIdx.x*blockDim.x;
    int j = threadIdx.y + blockIdx.y*blockDim.y;

    if (i >= nx || j >= nx) return;

    int s_i = threadIdx.x + 1;
    int s_j = threadIdx.y + 1;
    int s_ny = BLOCK_SIZE_Y + 2;

    // Load data into shared memory
    // Central square
    s_Un[getIndex(s_i, s_j, s_ny)] = Un[getIndex(i, j, ny)];
    // Top border
    if (s_i == 1 && s_j !=1 && i != 0)
    {
        s_Un[getIndex(0, s_j, s_ny)] = Un[getIndex(blockIdx.x*blockDim.x - 1, j, ny)];
    }
    // Bottom border
    if (s_i == BLOCK_SIZE_X && s_j != BLOCK_SIZE_Y && i != nx - 1)
    {
        s_Un[getIndex(BLOCK_SIZE_X + 1, s_j, s_ny)] = Un[getIndex((blockIdx.x + 1)*blockDim.x, j, ny)];
    }
    // Left border
    if (s_i != 1 && s_j == 1 && j != 0)
    {
        s_Un[getIndex(s_i, 0, s_ny)] = Un[getIndex(i, blockIdx.y*blockDim.y - 1, ny)];
    }
    // Right border
    if (s_i != BLOCK_SIZE_X && s_j == BLOCK_SIZE_Y && j != ny - 1)
    {
        s_Un[getIndex(s_i, BLOCK_SIZE_Y + 1, s_ny)] = Un[getIndex(i, (blockIdx.y + 1)*blockDim.y, ny)];
    }

    // Make sure all the data is loaded before computing
    __syncthreads();

    if (i > 0 && i < nx - 1)
    {
        if (j > 0 && j < ny - 1)
        {
            // double uij = s_Un[getIndex(s_i, s_j, s_ny)];
            double uim1j = s_Un[getIndex(s_i-1, s_j, s_ny)];
            double uijm1 = s_Un[getIndex(s_i, s_j-1, s_ny)];
            double uip1j = s_Un[getIndex(s_i+1, s_j, s_ny)];
            double uijp1 = s_Un[getIndex(s_i, s_j+1, s_ny)];

            // Explicit scheme
            Unp1[getIndex(i, j, ny)] = 0.25 * (uim1j + uijm1 + uip1j + uijp1);
        }
    }
}

int main()
{
    const int nx = 1024;   // Width of the area
    const int ny = 1024;   // Height of the area

    const double a = 0.5;     // Diffusion constant

    const double dx = 0.01;   // Horizontal grid spacing 
    const double dy = 0.01;   // Vertical grid spacing

    const double dx2 = dx*dx;
    const double dy2 = dy*dy;

    const double dt = dx2 * dy2 / (2.0 * a * (dx2 + dy2)); // Largest stable time step
    const int numSteps = 5000;                             // Number of time steps
    const int outputEvery = 1000;                          // How frequently to write output image

    int numElements = nx*ny;

    // Allocate two sets of data for current and next timesteps
    double* h_Un   = (double*)calloc(numElements, sizeof(double));

    // Initializing the data with a pattern of disk of radius of 1/6 of the width
    double radius2 = (nx/6.0) * (nx/6.0);
    for (int i = 0; i < nx; i++)
    {
        for (int j = 0; j < ny; j++)
        {
            int index = getIndex(i, j, ny);
            // Distance of point i, j from the origin
            double ds2 = (i - nx/2) * (i - nx/2) + (j - ny/2)*(j - ny/2);
            if (ds2 < radius2)
            {
                h_Un[index] = 65.0;
            }
            else
            {
                h_Un[index] = 5.0;
            }
        }
    }

    double* d_Un;
    double* d_Unp1;

    cudaMalloc((void**)&d_Un, numElements*sizeof(double));
    cudaMalloc((void**)&d_Unp1, numElements*sizeof(double));

    cudaMemcpy(d_Un, h_Un, numElements*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_Unp1, h_Un, numElements*sizeof(double), cudaMemcpyHostToDevice);

    dim3 numBlocks(nx/BLOCK_SIZE_X + 1, ny/BLOCK_SIZE_Y + 1);
    dim3 threadsPerBlock(BLOCK_SIZE_X, BLOCK_SIZE_Y);

    // Timing
    cudaEvent_t start, stop;
    float etime;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);
    // Main loop
    for (int n = 0; n <= numSteps; n++)
    {
        evolve_kernel<<<numBlocks, threadsPerBlock>>>(d_Un, d_Unp1, nx, ny, dx2, dy2, a*dt);

        // // Write the output if needed
        // if (n % outputEvery == 0)
        // {
        //     cudaMemcpy(h_Un, d_Un, numElements*sizeof(double), cudaMemcpyDeviceToHost);
        //     cudaError_t errorCode = cudaGetLastError();
        //     if (errorCode != cudaSuccess)
        //     {
        //         printf("Cuda error %d: %s\n", errorCode, cudaGetErrorString(errorCode));
        //         exit(0);
        //     }
        //     char filename[64];
        //     sprintf(filename, "heat_%04d.png", n);
        // }

        std::swap(d_Un, d_Unp1);
    }

    // Timing

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    cudaEventElapsedTime(&etime, start, stop);
    printf("time used %f", etime);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    // Release the memory
    free(h_Un);

    cudaFree(d_Un);
    cudaFree(d_Unp1);
    
    return 0;
}