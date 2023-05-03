
#include <iostream>
#include <fstream>
#include <string>
#include <cstdio>
#include <cstring>
#include <cassert>
#include <chrono>
#include <algorithm>
#include <vector>


using namespace std::chrono;
using namespace std;






#define N_THRD_PER_BLK 32
// #define N_BLK 512


#define BODY_SIZE_BYTE 64 
#define BODY_SIZE_WORD 16 
#define BATCH_SIZE (N_THRD_PER_BLK * 4 / BODY_SIZE_BYTE) // (32 * 4 / 64) == 2.
#define BATCH_SIZE_WORD (BATCH_SIZE * BODY_SIZE_WORD)

typedef unsigned int WORD;

struct Body{
    
    double qx, qy, qz, vx, vy, vz, m;
    long long isDevice;
    
};

struct Input{
    int n;
    int planetId;
    int asteroidId;
    Body *bodyArray;
};



__global__ void kernel_problem1(int step, int n, int planetId, int asteroidId,
                                                             unsigned char *bodyArray){

    int gtid = blockIdx.x * blockDim.x + threadIdx.x;
    int bodyId_this = gtid;
    int tid = threadIdx.x;
    if(bodyId_this >= n) return;


    double ax = 0, ay = 0, az = 0;
    double vx, vy, vz, qx, qy, qz;
    
    vx = ((Body *)bodyArray)[bodyId_this].vx;
    vy = ((Body *)bodyArray)[bodyId_this].vy;
    vz = ((Body *)bodyArray)[bodyId_this].vz;

    qx = ((Body *)bodyArray)[bodyId_this].qx;
    qy = ((Body *)bodyArray)[bodyId_this].qy;
    qz = ((Body *)bodyArray)[bodyId_this].qz;


    int n_batch = n / BATCH_SIZE;
    __shared__ WORD sm[BATCH_SIZE_WORD];

    for(int batchId = 0; batchId < n_batch; batchId++){
        
        if(batchId * BATCH_SIZE_WORD + tid < n * BODY_SIZE_WORD){
            sm[tid] = ((WORD *)bodyArray)[batchId * BATCH_SIZE_WORD + tid];
        }
        __syncthreads();


        for(int i = 0; i < BATCH_SIZE; i++){

            int bodyId_other = batchId * BATCH_SIZE + i;
            if(bodyId_other >= n) break;
            if (bodyId_other == bodyId_this) continue;


            double mj = ((Body *)sm)[i].m;
            if (((Body *)sm)[i].isDevice == 1) {
                mj = gravity_device_mass(mj, step * dt);
            }

            double dx = ((Body *)sm)[i].qx - qx;
            double dy = ((Body *)sm)[i].qy - qy;
            double dz = ((Body *)sm)[i].qz - qz;

            double dist3 = pow(dx * dx + dy * dy + dz * dz + eps *eps, 1.5);

            ax += G * mj * dx / dist3;    
            ay += G * mj * dy / dist3;    
            az += G * mj * dz / dist3; 

        }
    }


    vx += ax * param::dt;
    vy += ay * param::dt;
    vz += az * param::dt;
    
    qx += vx * param::dt;
    qy += vy * param::dt;
    qz += vz * param::dt; 
}





void read_input(const char* filename, Input *input) {

    std::ifstream fin(filename);
    fin >> input->n >> input->planetId >> input->asteroidId;

    input->bodyArray = new Body[input->n];

    string type;

    for (int i = 0; i < input->n; i++) {
        fin >> input->bodyArray[i].qx 
            >> input->bodyArray[i].qy
            >> input->bodyArray[i].qz 
            >> input->bodyArray[i].vx 
            >> input->bodyArray[i].vy 
            >> input->bodyArray[i].vz 
            >> input->bodyArray[i].m 
            >> type;
        
        if (type != "device"){
            input->bodyArray[i].isDevice = 0;
        }
        else{
            input->bodyArray[i].isDevice = 1;
        }
    }

}




int main(int argc, char **argv)
{
    Input input;
    unsigned char *bodyArray_dev;

    read_input(argv[1], &input);

    for (int i = 0; i < input.n; i++) {
        if (input.bodyArray[i].isDevice == 1) input.bodyArray[i].m = 0;
    }

    printf("sizeof(Body): %d\n", sizeof(Body));

    cudaSetDevice(0);
    cudaMalloc(&bodyArray_dev, input.n * sizeof(Body));
    cudaMemcpy(bodyArray_dev, (unsigned char *)input.bodyArray,
                                          input.n * sizeof(Body), cudaMemcpyHostToDevice);
    
    for (int step = 1; step <= param::n_steps; step++) {

        int n_block = input.n / N_THRD_PER_BLK + 1;

        kernel_problem1<<<n_block, N_THRD_PER_BLK>>>(step, input.n, input.planetId, 
                                            input.asteroidId, bodyArray_dev);

    }
















    // Problem 1

    // int size = N_BLK * N_THRD_PER_BLK;
    // int *devSrc0, *hostSrc0;
    // hostSrc0 = new int[size];
    // for (int i = 0 ;i<size;i++){
    //     hostSrc0[i] = 0;
    // }

    // cudaSetDevice(0);
    // cudaMalloc(&devSrc0, size * sizeof(int));
    // cudaMemcpy((unsigned char *)devSrc0, (unsigned char *)hostSrc0,
    //                                      size * sizeof(int), cudaMemcpyHostToDevice);

    // kernel<<<N_BLK, N_THRD_PER_BLK, 0>>>(devSrc0, 4);


    // cudaDeviceSynchronize();   





    return 0;
}