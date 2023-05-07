
#include <iostream>
#include <fstream>
#include <string>
#include <cstdio>
#include <cstring>
#include <cassert>
#include <chrono>
#include <algorithm>
#include <vector>



// const int n_steps = 200000;
// const double dt = 60;
// const double eps = 1e-3;
// const double G = 6.674e-11;

#define n_steps 200000
#define dt 60
#define eps 1e-6
#define G 6.674e-11
#define planet_radius 1e7
#define missile_speed 1e6


__device__ 
double gravity_device_mass(double m0, double t) {
    return m0 + 0.5 * m0 * fabs(sin(t / 6000));
}

__device__
double get_missile_cost(double t) {
    return 1e5 + 1e3 * t; 
}



using namespace std::chrono;
using namespace std;

#define N_THRD_PER_BLK_X 4
#define N_THRD_PER_BLK_Y 32
#define N_THRD_PER_BLK (N_THRD_PER_BLK_X * N_THRD_PER_BLK_Y)


#define BODY_SIZE_BYTE 64 
#define BODY_SIZE_WORD 16 
#define BATCH_SIZE (N_THRD_PER_BLK_Y)
#define BATCH_SIZE_WORD (BATCH_SIZE * BODY_SIZE_WORD)

// need to make sure that this is int.
#define N_BODY_COPY_PER_PASS (N_THRD_PER_BLK * 4 / BODY_SIZE_BYTE) // (32 * 4 / 64) == 2.


typedef unsigned int WORD;
typedef unsigned char BYTE;

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



__global__ void kernel_problem1(int step, int n_batch, int n, int planetId, int asteroidId,
                                BYTE *bodyArray, BYTE *bodyArray_update, BYTE *min_dist_sq){

    int bodyId_this = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.y * blockDim.x + threadIdx.x;

    double ax = 0, ay = 0, az = 0, dx, dy, dz;
    double vx, vy, vz, qx, qy, qz;
   
    if(bodyId_this < n){
       
        qx = ((Body *)bodyArray)[bodyId_this].qx;
        qy = ((Body *)bodyArray)[bodyId_this].qy;
        qz = ((Body *)bodyArray)[bodyId_this].qz;
    }

    // update min_dist.
    if((bodyId_this == planetId) && (threadIdx.y == 0)){

        dx = qx - ((Body *)bodyArray)[asteroidId].qx;
        dy = qy - ((Body *)bodyArray)[asteroidId].qy;
        dz = qz - ((Body *)bodyArray)[asteroidId].qz;

        *((double *)min_dist_sq) = min(*((double *)min_dist_sq), 
                                             dx * dx + dy * dy + dz * dz);  
    }

    __shared__ WORD sm[BATCH_SIZE_WORD + N_THRD_PER_BLK_Y * 3 * 2 * N_THRD_PER_BLK_X];
    double *sm_aggregate = (double *)(sm + BATCH_SIZE_WORD);


    for(int batchId = 0; batchId < n_batch; batchId++){

        for(int i = 0; i < BATCH_SIZE; i += N_BODY_COPY_PER_PASS){

            int global_offset = batchId * BATCH_SIZE_WORD;
            int local_offset = i * BODY_SIZE_WORD + tid;
            int idx = global_offset + local_offset;

            if(idx < n * BODY_SIZE_WORD){
                sm[local_offset] = ((WORD *)bodyArray)[idx];
            }
        }

        __syncthreads();

        int bodyId_other = batchId * BATCH_SIZE + threadIdx.y;
        
        if ((bodyId_other != bodyId_this) && (bodyId_other < n)){
            
            dx = ((Body *)sm)[threadIdx.y].qx - qx;
            dy = ((Body *)sm)[threadIdx.y].qy - qy;
            dz = ((Body *)sm)[threadIdx.y].qz - qz;

            double mj = ((Body *)sm)[threadIdx.y].m;
     
            if (((Body *)sm)[threadIdx.y].isDevice == 1) {
                mj = gravity_device_mass(mj, step * dt);
            }

            // double dist3 = pow(dx * dx + dy * dy + dz * dz + eps, 1.5);
            double dist2 = dx * dx + dy * dy + dz * dz + eps;
            double dist3 = sqrt(dist2) * dist2;



            ax += G * mj * dx / dist3;    
            ay += G * mj * dy / dist3;    
            az += G * mj * dz / dist3; 

       

        }
    }

    sm_aggregate[threadIdx.y * (3 * blockDim.x) + 3 * threadIdx.x + 0] = ax * dt;
    sm_aggregate[threadIdx.y * (3 * blockDim.x) + 3 * threadIdx.x + 1] = ay * dt;
    sm_aggregate[threadIdx.y * (3 * blockDim.x) + 3 * threadIdx.x + 2] = az * dt;

    __syncthreads();

   
                      
    for(int binSize = 2; binSize <= blockDim.y; binSize = binSize << 1){

        if((threadIdx.y & (binSize - 1)) == 0){

            sm_aggregate[threadIdx.y * (3 * blockDim.x) + 3 * threadIdx.x + 0] += \
                sm_aggregate[(threadIdx.y + (binSize >> 1)) * (3 * blockDim.x) \
                        + 3 * threadIdx.x + 0];
            
            sm_aggregate[threadIdx.y * (3 * blockDim.x) + 3 * threadIdx.x + 1] += \
                sm_aggregate[(threadIdx.y + (binSize >> 1)) * (3 * blockDim.x) \
                        + 3 * threadIdx.x + 1];
            
            sm_aggregate[threadIdx.y * (3 * blockDim.x) + 3 * threadIdx.x + 2] += \
                sm_aggregate[(threadIdx.y + (binSize >> 1)) * (3 * blockDim.x) \
                        + 3 * threadIdx.x + 2];

        }
        __syncthreads();
    }

    
    __syncthreads();


    if(threadIdx.y == 0){

        vx = ((Body *)bodyArray)[bodyId_this].vx;
        vy = ((Body *)bodyArray)[bodyId_this].vy;
        vz = ((Body *)bodyArray)[bodyId_this].vz;

        vx += sm_aggregate[3 * threadIdx.x + 0];
        vy += sm_aggregate[3 * threadIdx.x + 1];
        vz += sm_aggregate[3 * threadIdx.x + 2];

        qx += vx * dt;
        qy += vy * dt;
        qz += vz * dt; 

        // write back.
        if(bodyId_this < n){
            ((Body *)bodyArray_update)[bodyId_this].vx = vx;
            ((Body *)bodyArray_update)[bodyId_this].vy = vy;
            ((Body *)bodyArray_update)[bodyId_this].vz = vz;

            ((Body *)bodyArray_update)[bodyId_this].qx = qx;
            ((Body *)bodyArray_update)[bodyId_this].qy = qy;
            ((Body *)bodyArray_update)[bodyId_this].qz = qz; 
        }  
    }
}





int main(int argc, char **argv)
{
    Input input;
    BYTE *bodyArray1_dev, *bodyArray2_dev, *min_dist_sq_dev;




    read_input(argv[1], &input);

    for (int i = 0; i < input.n; i++) {
        if (input.bodyArray[i].isDevice == 1) input.bodyArray[i].m = 0;
    }

    cudaSetDevice(0);

    cudaMalloc(&bodyArray1_dev, input.n * sizeof(Body));
    cudaMemcpy(bodyArray1_dev, (BYTE *)(input.bodyArray),
                            input.n * sizeof(Body), cudaMemcpyHostToDevice);

    cudaMalloc(&bodyArray2_dev, input.n * sizeof(Body));
    cudaMemcpy(bodyArray2_dev, (BYTE *)(input.bodyArray),
                            input.n * sizeof(Body), cudaMemcpyHostToDevice);

    double min_dist_sq_host = std::numeric_limits<double>::infinity();

    cudaMalloc(&min_dist_sq_dev, sizeof(double));
    cudaMemcpy(min_dist_sq_dev, (BYTE *)&min_dist_sq_host,
                                    sizeof(double), cudaMemcpyHostToDevice);

    int n_block = input.n / N_THRD_PER_BLK_X + 1;
    dim3 nThreadsPerBlock(N_THRD_PER_BLK_X, N_THRD_PER_BLK_Y, 1);

    int n_batch = input.n / BATCH_SIZE;
    if(n_batch * BATCH_SIZE < input.n) n_batch += 1;

    auto start = high_resolution_clock::now();

    for (int step = 1; step <= n_steps + 1; step++) {

        kernel_problem1<<<n_block, nThreadsPerBlock>>>(step, n_batch, input.n, input.planetId, 
                     input.asteroidId, bodyArray1_dev, bodyArray2_dev, min_dist_sq_dev);
        
        BYTE *tmp = bodyArray1_dev;
        bodyArray1_dev = bodyArray2_dev;
        bodyArray2_dev = tmp;
    }

    cudaDeviceSynchronize();

    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(stop - start);
    cout<<"problem 1 time: "<<duration.count() / 1000000. <<" sec"<<endl;

    cudaMemcpy((BYTE *)&min_dist_sq_host, min_dist_sq_dev, 
                                    sizeof(double), cudaMemcpyDeviceToHost);    

    printf("min_dist_host: %f\n", sqrt(min_dist_sq_host));
    












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