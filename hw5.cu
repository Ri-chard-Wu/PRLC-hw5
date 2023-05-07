
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
#define eps 1e-3
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

// // 34 sec
// #define N_THRD_PER_BLK_X 4
// #define N_THRD_PER_BLK_Y 32

// // 54 sec
// #define N_THRD_PER_BLK_X 4
// #define N_THRD_PER_BLK_Y 64

// // 37 sec
// #define N_THRD_PER_BLK_X 5
// #define N_THRD_PER_BLK_Y 32

// // 41 sec
// #define N_THRD_PER_BLK_X 4
// #define N_THRD_PER_BLK_Y 16

#define N_THRD_PER_BLK_X 16
#define N_THRD_PER_BLK_Y 8
#define N_THRD_PER_BLK (N_THRD_PER_BLK_X * N_THRD_PER_BLK_Y)


#define BODY_SIZE_BYTE 64 
#define BODY_SIZE_WORD 16 
#define BATCH_SIZE (N_THRD_PER_BLK_X)
#define BATCH_SIZE_WORD (BATCH_SIZE * BODY_SIZE_WORD)

// need to make sure that this is int.
#define N_BODY_COPY_PER_PASS (N_THRD_PER_BLK * 4 / BODY_SIZE_BYTE) // (32 * 4 / 64) == 2.


typedef unsigned int WORD;
typedef unsigned char BYTE;

// struct Body{
    
//     double qx, qy, qz, vx, vy, vz, m;
//     long long isDevice;
    
// };


struct BodyArray{
    
    double *qx, *qy, *qz, *vx, *vy, *vz, *m;
    int *isDevice;
    
};


struct Input{
    int n;
    int planetId;
    int asteroidId;
    BodyArray bodyArray;
};



void read_input(const char* filename, Input *input) {

    std::ifstream fin(filename);
    fin >> input->n >> input->planetId >> input->asteroidId;

    input->bodyArray.qx = new double[input->n];
    input->bodyArray.qy = new double[input->n];
    input->bodyArray.qz = new double[input->n];
    input->bodyArray.vx = new double[input->n];
    input->bodyArray.vy = new double[input->n];
    input->bodyArray.vz = new double[input->n];
    input->bodyArray.m = new double[input->n];
    input->bodyArray.isDevice = new int[input->n];

    string type;

    for (int i = 0; i < input->n; i++) {
        fin >> input->bodyArray.qx[i]
            >> input->bodyArray.qy[i]
            >> input->bodyArray.qz[i]
            >> input->bodyArray.vx[i]
            >> input->bodyArray.vy[i]
            >> input->bodyArray.vz[i]
            >> input->bodyArray.m[i]
            >> type;
        
        if (type != "device"){
            input->bodyArray.isDevice[i] = 0;
        }
        else{
            input->bodyArray.isDevice[i] = 1;
        }
    }
}




__global__ void kernel_problem1(int step, int n, int planetId, int asteroidId,
                BodyArray *bodyArray, BodyArray *bodyArray_update, BYTE *min_dist_sq){


    int bodyId_this = blockIdx.x * blockDim.y + threadIdx.y;
    // int tid = threadIdx.y * blockDim.x + threadIdx.x;

    double ax = 0, ay = 0, az = 0, dx, dy, dz;
    double qx, qy, qz;
    
    if(bodyId_this < n){
        qx = bodyArray->qx[bodyId_this];
        qy = bodyArray->qy[bodyId_this];
        qz = bodyArray->qz[bodyId_this];
    }

    // update min_dist.
    if((bodyId_this == planetId) && (threadIdx.x == 0)){

        dx = qx - bodyArray->qx[asteroidId];
        dy = qy - bodyArray->qy[asteroidId];
        dz = qz - bodyArray->qz[asteroidId];

        *((double *)min_dist_sq) = min(*((double *)min_dist_sq), 
                                             dx * dx + dy * dy + dz * dz);  
    }

    __shared__ double sm[BATCH_SIZE_WORD + N_THRD_PER_BLK_Y * 3 * 2 * N_THRD_PER_BLK_X];
    double *sm_double = (double *)sm;
    int *sm_int = (int *)sm;
    double *sm_aggregate = (double *)(sm + BATCH_SIZE_WORD);



    int n_batch = n / BATCH_SIZE;
    if(n_batch * BATCH_SIZE < n) n_batch += 1;

    for(int batchId = 0; batchId < n_batch; batchId++){

        if(batchId * BATCH_SIZE + threadIdx.x < n){

            sm_double[0 * BATCH_SIZE + threadIdx.x] = bodyArray->qx[batchId * BATCH_SIZE + threadIdx.x];
            sm_double[1 * BATCH_SIZE + threadIdx.x] = bodyArray->qy[batchId * BATCH_SIZE + threadIdx.x];
            sm_double[2 * BATCH_SIZE + threadIdx.x] = bodyArray->qz[batchId * BATCH_SIZE + threadIdx.x];
            sm_double[3 * BATCH_SIZE + threadIdx.x] = bodyArray->m[batchId * BATCH_SIZE + threadIdx.x];
            sm_int[8 * BATCH_SIZE + threadIdx.x] = bodyArray->isDevice[batchId * BATCH_SIZE + threadIdx.x];
        }



        __syncthreads();

        int bodyId_other = batchId * BATCH_SIZE + threadIdx.x;
        
        if ((bodyId_other != bodyId_this) && (bodyId_other < n)){
            
            double mj = sm_double[3 * BATCH_SIZE + threadIdx.x];
     
            if (sm_int[8 * BATCH_SIZE + threadIdx.x] == 1) {
                mj = gravity_device_mass(mj, step * dt);
            }

            dx = sm_double[0 * BATCH_SIZE + threadIdx.x] - qx;
            dy = sm_double[1 * BATCH_SIZE + threadIdx.x] - qy;
            dz = sm_double[2 * BATCH_SIZE + threadIdx.x] - qz;

            double dist3 = pow(dx * dx + dy * dy + dz * dz + eps * eps, 1.5);

            ax += G * mj * dx / dist3;    
            ay += G * mj * dy / dist3;    
            az += G * mj * dz / dist3; 
        }
    }



    sm_aggregate[threadIdx.x * (3 * blockDim.y) + 3 * threadIdx.y + 0] = ax * dt;
    sm_aggregate[threadIdx.x * (3 * blockDim.y) + 3 * threadIdx.y + 1] = ay * dt;
    sm_aggregate[threadIdx.x * (3 * blockDim.y) + 3 * threadIdx.y + 2] = az * dt;

    // sm_aggregate[0 * blockDim.x * blockDim.y + (threadIdx.y * blockDim.x + threadIdx.x)] = ax * dt;
    // sm_aggregate[1 * blockDim.x * blockDim.y + (threadIdx.y * blockDim.x + threadIdx.x)] = ay * dt;
    // sm_aggregate[2 * blockDim.x * blockDim.y + (threadIdx.y * blockDim.x + threadIdx.x)] = az * dt;
    
    __syncthreads();

   
                      
    for(int binSize = 2; binSize <= blockDim.x; binSize = binSize << 1){

        if((threadIdx.x & (binSize - 1)) == 0){

            sm_aggregate[threadIdx.x * (3 * blockDim.y) + 3 * threadIdx.y + 0] += \
                sm_aggregate[(threadIdx.x + (binSize >> 1)) * (3 * blockDim.y) \
                        + 3 * threadIdx.y + 0];
            
            sm_aggregate[threadIdx.x * (3 * blockDim.y) + 3 * threadIdx.y + 1] += \
                sm_aggregate[(threadIdx.x + (binSize >> 1)) * (3 * blockDim.y) \
                        + 3 * threadIdx.y + 1];

            sm_aggregate[threadIdx.x * (3 * blockDim.y) + 3 * threadIdx.y + 2] += \
                sm_aggregate[(threadIdx.x + (binSize >> 1)) * (3 * blockDim.y) \
                        + 3 * threadIdx.y + 2];

        }
        __syncthreads();
    }



    if((threadIdx.x == 0) && (bodyId_this < n)){

        double vx = bodyArray->vx[bodyId_this];
        double vy = bodyArray->vy[bodyId_this];
        double vz = bodyArray->vz[bodyId_this];
        
        vx += sm_aggregate[3 * threadIdx.y + 0];
        vy += sm_aggregate[3 * threadIdx.y + 1];
        vz += sm_aggregate[3 * threadIdx.y + 2];

        bodyArray_update->vx[bodyId_this] = vx;
        bodyArray_update->vy[bodyId_this] = vy;
        bodyArray_update->vz[bodyId_this] = vz;

        bodyArray_update->qx[bodyId_this] = qx + vx * dt;
        bodyArray_update->qy[bodyId_this] = qy + vy * dt;
        bodyArray_update->qz[bodyId_this] = qz + vz * dt;
    }
}







void problem1(cudaStream_t stream, char* filename, double *min_dist_sq_ptr){

    Input input;
    read_input(filename, &input);
    for (int i = 0; i < input.n; i++) {
        if (input.bodyArray.isDevice[i] == 1) input.bodyArray.m[i] = 0;
    }



    BYTE *min_dist_sq_dev;

    BodyArray *bodyArray1_dev, *bodyArray2_dev;
    BodyArray bodyArray_host1, bodyArray_host2;
    
    // ----------------------------------------

    cudaMalloc(&(bodyArray_host1.qx), input.n * sizeof(double));
    cudaMalloc(&(bodyArray_host1.qy), input.n * sizeof(double));
    cudaMalloc(&(bodyArray_host1.qz), input.n * sizeof(double));
    cudaMalloc(&(bodyArray_host1.vx), input.n * sizeof(double));
    cudaMalloc(&(bodyArray_host1.vy), input.n * sizeof(double));
    cudaMalloc(&(bodyArray_host1.vz), input.n * sizeof(double));
    cudaMalloc(&(bodyArray_host1.m), input.n * sizeof(double));
    cudaMalloc(&(bodyArray_host1.isDevice), input.n * sizeof(int));

    cudaMemcpy((BYTE *)(bodyArray_host1.qx), (BYTE *)(input.bodyArray.qx), input.n * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy((BYTE *)(bodyArray_host1.qy), (BYTE *)(input.bodyArray.qy), input.n * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy((BYTE *)(bodyArray_host1.qz), (BYTE *)(input.bodyArray.qz), input.n * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy((BYTE *)(bodyArray_host1.vx), (BYTE *)(input.bodyArray.vx), input.n * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy((BYTE *)(bodyArray_host1.vy), (BYTE *)(input.bodyArray.vy), input.n * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy((BYTE *)(bodyArray_host1.vz), (BYTE *)(input.bodyArray.vz), input.n * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy((BYTE *)(bodyArray_host1.m), (BYTE *)(input.bodyArray.m), input.n * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy((BYTE *)(bodyArray_host1.isDevice), (BYTE *)(input.bodyArray.isDevice), input.n * sizeof(int), cudaMemcpyHostToDevice);

    cudaMalloc(&bodyArray1_dev, sizeof(BodyArray));

    cudaMemcpy((BYTE *)bodyArray1_dev, (BYTE *)(&bodyArray_host1),
                            sizeof(BodyArray), cudaMemcpyHostToDevice);
    // ----------------------------------------


    cudaMalloc(&(bodyArray_host2.qx), input.n * sizeof(double));
    cudaMalloc(&(bodyArray_host2.qy), input.n * sizeof(double));
    cudaMalloc(&(bodyArray_host2.qz), input.n * sizeof(double));
    cudaMalloc(&(bodyArray_host2.vx), input.n * sizeof(double));
    cudaMalloc(&(bodyArray_host2.vy), input.n * sizeof(double));
    cudaMalloc(&(bodyArray_host2.vz), input.n * sizeof(double));
    cudaMalloc(&(bodyArray_host2.m), input.n * sizeof(double));
    cudaMalloc(&(bodyArray_host2.isDevice), input.n * sizeof(int));

    cudaMemcpy((BYTE *)(bodyArray_host2.qx), (BYTE *)(input.bodyArray.qx), input.n * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy((BYTE *)(bodyArray_host2.qy), (BYTE *)(input.bodyArray.qy), input.n * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy((BYTE *)(bodyArray_host2.qz), (BYTE *)(input.bodyArray.qz), input.n * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy((BYTE *)(bodyArray_host2.vx), (BYTE *)(input.bodyArray.vx), input.n * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy((BYTE *)(bodyArray_host2.vy), (BYTE *)(input.bodyArray.vy), input.n * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy((BYTE *)(bodyArray_host2.vz), (BYTE *)(input.bodyArray.vz), input.n * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy((BYTE *)(bodyArray_host2.m), (BYTE *)(input.bodyArray.m), input.n * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy((BYTE *)(bodyArray_host2.isDevice), (BYTE *)(input.bodyArray.isDevice), input.n * sizeof(int), cudaMemcpyHostToDevice);


    cudaMalloc(&bodyArray2_dev, sizeof(BodyArray));
    
    cudaMemcpy((BYTE *)bodyArray2_dev, (BYTE *)(&bodyArray_host2),
                            sizeof(BodyArray), cudaMemcpyHostToDevice);


    // ----------------------------------------

    double min_dist_sq_host = std::numeric_limits<double>::infinity();
    double min_dist_host;

    cudaMalloc(&min_dist_sq_dev, sizeof(double));
    cudaMemcpyAsync(min_dist_sq_dev, (BYTE *)&min_dist_sq_host,
                                    sizeof(double), cudaMemcpyHostToDevice);

    int n_block = input.n / N_THRD_PER_BLK_Y + 1;
    dim3 nThreadsPerBlock(N_THRD_PER_BLK_X, N_THRD_PER_BLK_Y, 1);

    // printf("[host] qx, qy, qz: %f, %f, %f\n", input.bodyArray.m[10], 
    //                                           input.bodyArray.vy[10], 
    //                                           input.bodyArray.vz[10]);

    auto start = high_resolution_clock::now();

    for (int step = 1; step <= n_steps + 1; step++) {

        kernel_problem1<<<n_block, nThreadsPerBlock, 0, stream>>>\
                (step, input.n, input.planetId, input.asteroidId, 
                             bodyArray1_dev, bodyArray2_dev, min_dist_sq_dev);
        
        BodyArray *tmp = bodyArray1_dev;
        bodyArray1_dev = bodyArray2_dev;
        bodyArray2_dev = tmp;
    }


    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(stop - start);
    cout<<"problem 1 time: "<<duration.count() / 1000000. <<" sec"<<endl;

    cudaMemcpyAsync((BYTE *)min_dist_sq_ptr, min_dist_sq_dev, 
                                    sizeof(double), cudaMemcpyDeviceToHost);    

}







int main(int argc, char **argv)
{

    cudaSetDevice(0);
    cudaStream_t stream0[2];
    for (int i = 0; i < 2; ++i) cudaStreamCreate(&stream0[i]);
    
    cudaSetDevice(1);
    cudaStream_t stream1[2];
    for (int i = 0; i < 2; ++i) cudaStreamCreate(&stream1[i]);
    




    cudaSetDevice(0);
    double min_dist, min_dist_sq;
    problem1(stream0[0], argv[1], &min_dist_sq);
    cudaDeviceSynchronize();
    
    min_dist = sqrt(min_dist_sq);
    printf("min_dist: %f\n", min_dist);






  





    return 0;
}