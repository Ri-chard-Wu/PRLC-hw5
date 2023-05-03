
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


#define N_THRD_PER_BLK 32

#define BODY_SIZE_BYTE 64 
#define BODY_SIZE_WORD 16 
#define BATCH_SIZE (N_THRD_PER_BLK * 4 / BODY_SIZE_BYTE) // (32 * 4 / 64) == 2.
#define BATCH_SIZE_WORD (BATCH_SIZE * BODY_SIZE_WORD)

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





__global__ void kernel_problem1(int step, int n, int planetId, int asteroidId,
                                            BYTE *bodyArray, BYTE *min_dist){

                                               

    int gtid = blockIdx.x * blockDim.x + threadIdx.x;
    int bodyId_this = gtid;
    int tid = threadIdx.x;
    // if(bodyId_this >= n) return;



    // if(step == 1 && gtid == 0){
    //     printf("*((double *)min_dist): %f\n", *((double *)min_dist)); 
    // }


    double ax = 0, ay = 0, az = 0, dx, dy, dz;
    double vx, vy, vz, qx, qy, qz;

    if(bodyId_this < n){
        vx = ((Body *)bodyArray)[bodyId_this].vx;
        vy = ((Body *)bodyArray)[bodyId_this].vy;
        vz = ((Body *)bodyArray)[bodyId_this].vz;

        qx = ((Body *)bodyArray)[bodyId_this].qx;
        qy = ((Body *)bodyArray)[bodyId_this].qy;
        qz = ((Body *)bodyArray)[bodyId_this].qz;
    }



    int n_batch = n / BATCH_SIZE;
    if(n_batch * BATCH_SIZE < n) n_batch += 1;
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

            dx = ((Body *)sm)[i].qx - qx;
            dy = ((Body *)sm)[i].qy - qy;
            dz = ((Body *)sm)[i].qz - qz;

            double dist3 = pow(dx * dx + dy * dy + dz * dz + eps *eps, 1.5);

            ax += G * mj * dx / dist3;    
            ay += G * mj * dy / dist3;    
            az += G * mj * dz / dist3; 
        }
    }



    vx += ax * dt;
    vy += ay * dt;
    vz += az * dt;
    
    qx += vx * dt;
    qy += vy * dt;
    qz += vz * dt; 


    // write back.
    if(bodyId_this < n){
        ((Body *)bodyArray)[bodyId_this].vx = vx;
        ((Body *)bodyArray)[bodyId_this].vy = vy;
        ((Body *)bodyArray)[bodyId_this].vz = vz;

        ((Body *)bodyArray)[bodyId_this].qx = qx;
        ((Body *)bodyArray)[bodyId_this].qy = qy;
        ((Body *)bodyArray)[bodyId_this].qz = qz; 
    }   

    __syncthreads();

    // update min_dist.
    if(bodyId_this == planetId){

        dx = qx - ((Body *)bodyArray)[asteroidId].qx;
        dy = qy - ((Body *)bodyArray)[asteroidId].qy;
        dz = qz - ((Body *)bodyArray)[asteroidId].qz;

        *((double *)min_dist) = min(*((double *)min_dist), 
                                sqrt(dx * dx + dy * dy + dz * dz));  
    }
}





int main(int argc, char **argv)
{
    Input input;
    BYTE *bodyArray_dev, *min_dist_dev;


    auto start = high_resolution_clock::now();



    read_input(argv[1], &input);

    for (int i = 0; i < input.n; i++) {
        if (input.bodyArray[i].isDevice == 1) input.bodyArray[i].m = 0;
    }

    // printf("sizeof(Body): %d\n", sizeof(Body));


    cudaSetDevice(0);

    cudaMalloc(&bodyArray_dev, input.n * sizeof(Body));
    cudaMemcpy(bodyArray_dev, (BYTE *)(input.bodyArray),
                            input.n * sizeof(Body), cudaMemcpyHostToDevice);
    
    double dx = input.bodyArray[input.planetId].qx - input.bodyArray[input.asteroidId].qx;
    double dy = input.bodyArray[input.planetId].qy - input.bodyArray[input.asteroidId].qy;
    double dz = input.bodyArray[input.planetId].qz - input.bodyArray[input.asteroidId].qz;
    double min_dist_host = sqrt(dx * dx + dy * dy + dz * dz);
    // printf("min_dist_host: %f\n", min_dist_host);

    cudaMalloc(&min_dist_dev, sizeof(double));
    cudaMemcpy(min_dist_dev, (BYTE *)&min_dist_host,
                                    sizeof(double), cudaMemcpyHostToDevice);

    int n_block = input.n / N_THRD_PER_BLK + 1;
    
    for (int step = 1; step <= n_steps; step++) {
        kernel_problem1<<<n_block, N_THRD_PER_BLK>>>(step, input.n, input.planetId, 
                                            input.asteroidId, bodyArray_dev, min_dist_dev);                                          
    }

    cudaDeviceSynchronize();
    cudaMemcpy((BYTE *)&min_dist_host, min_dist_dev, 
                                    sizeof(double), cudaMemcpyDeviceToHost);    

    printf("min_dist_host: %f\n", min_dist_host);
    
    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(stop - start);
    cout<<"problem 1 time: "<<duration.count()/ 1000000.0 <<" sec"<<endl;











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