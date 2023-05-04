
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
                                Body *bodyArray, Body *bodyArray_update, BYTE *min_dist_sq){


    int bodyId_this = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.y * blockDim.x + threadIdx.x;

    double ax = 0, ay = 0, az = 0, dx, dy, dz;
    double qx, qy, qz;

    
    // clock_t start_time = clock(); 
    // clock_t stop_time = clock();
    // int runtime = (int)(stop_time - start_time);

   
    if(bodyId_this < n){
        qx = bodyArray[bodyId_this].qx;
        qy = bodyArray[bodyId_this].qy;
        qz = bodyArray[bodyId_this].qz;
    }

    // update min_dist.
    if((bodyId_this == planetId) && (threadIdx.y == 0)){

        dx = qx - bodyArray[asteroidId].qx;
        dy = qy - bodyArray[asteroidId].qy;
        dz = qz - bodyArray[asteroidId].qz;

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
            
            double mj = ((Body *)sm)[threadIdx.y].m;
     
            if (((Body *)sm)[threadIdx.y].isDevice == 1) {
                mj = gravity_device_mass(mj, step * dt);
            }

            dx = ((Body *)sm)[threadIdx.y].qx - qx;
            dy = ((Body *)sm)[threadIdx.y].qy - qy;
            dz = ((Body *)sm)[threadIdx.y].qz - qz;

            // if(bodyId_this == 0) start_time = clock();

            double dist3 = pow(dx * dx + dy * dy + dz * dz + eps * eps, 1.5);

            ax += G * mj * dx / dist3;    
            ay += G * mj * dy / dist3;    
            az += G * mj * dz / dist3; 

            // if(bodyId_this == 0){
            //     stop_time = clock();
            //     runtime = (int)(stop_time - start_time);
            //     printf("dt: %d\n", runtime);
            // } 
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



    if((threadIdx.y < 3) && (bodyId_this < n)){
        
        double *v_ptr = (double *)&(bodyArray[bodyId_this].vx);
        double *q_ptr_update = (double *)&(bodyArray_update[bodyId_this].qx);
        double *v_ptr_update = (double *)&(bodyArray_update[bodyId_this].vx);
        double *q_ptr_update_sm = sm_aggregate + (3 * blockDim.x + 3 * threadIdx.x);
   
        q_ptr_update_sm[0] = qx;
        q_ptr_update_sm[1] = qy;
        q_ptr_update_sm[2] = qz;
        
        double vi = v_ptr[threadIdx.y];
        vi += sm_aggregate[3 * threadIdx.x + threadIdx.y];

        v_ptr_update[threadIdx.y] = vi;
        q_ptr_update[threadIdx.y] = q_ptr_update_sm[threadIdx.y] + vi * dt;
        
    }
}







__global__ void kernel_problem2(int step, int n_batch, int n, int planetId, int asteroidId,
                                Body *bodyArray, Body *bodyArray_update, BYTE *hit_time_step){


    int bodyId_this = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.y * blockDim.x + threadIdx.x;

    double ax = 0, ay = 0, az = 0, dx, dy, dz;
    double qx, qy, qz;

    
    // clock_t start_time = clock(); 
    // clock_t stop_time = clock();
    // int runtime = (int)(stop_time - start_time);

   
    if(bodyId_this < n){
        qx = bodyArray[bodyId_this].qx;
        qy = bodyArray[bodyId_this].qy;
        qz = bodyArray[bodyId_this].qz;
    }


    if((bodyId_this == planetId) && (threadIdx.y == 0)){

        dx = qx - bodyArray[asteroidId].qx;
        dy = qy - bodyArray[asteroidId].qy;
        dz = qz - bodyArray[asteroidId].qz;
        if (dx * dx + dy * dy + dz * dz < planet_radius * planet_radius) {
            if(*((int *)hit_time_step) == -2){
                *((int *)hit_time_step) = step - 1;
            }
        }
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
            
            double mj = ((Body *)sm)[threadIdx.y].m;
     
            if (((Body *)sm)[threadIdx.y].isDevice == 1) {
                mj = gravity_device_mass(mj, step * dt);
            }

            dx = ((Body *)sm)[threadIdx.y].qx - qx;
            dy = ((Body *)sm)[threadIdx.y].qy - qy;
            dz = ((Body *)sm)[threadIdx.y].qz - qz;

            // if(bodyId_this == 0) start_time = clock();

            double dist3 = pow(dx * dx + dy * dy + dz * dz + eps * eps, 1.5);

            ax += G * mj * dx / dist3;    
            ay += G * mj * dy / dist3;    
            az += G * mj * dz / dist3; 

            // if(bodyId_this == 0){
            //     stop_time = clock();
            //     runtime = (int)(stop_time - start_time);
            //     printf("dt: %d\n", runtime);
            // } 
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



    if((threadIdx.y < 3) && (bodyId_this < n)){
        
        double *v_ptr = (double *)&(bodyArray[bodyId_this].vx);
        double *q_ptr_update = (double *)&(bodyArray_update[bodyId_this].qx);
        double *v_ptr_update = (double *)&(bodyArray_update[bodyId_this].vx);
        double *q_ptr_update_sm = sm_aggregate + (3 * blockDim.x + 3 * threadIdx.x);
   
        q_ptr_update_sm[0] = qx;
        q_ptr_update_sm[1] = qy;
        q_ptr_update_sm[2] = qz;
        
        double vi = v_ptr[threadIdx.y];
        vi += sm_aggregate[3 * threadIdx.x + threadIdx.y];

        v_ptr_update[threadIdx.y] = vi;
        q_ptr_update[threadIdx.y] = q_ptr_update_sm[threadIdx.y] + vi * dt;
        
    }
}





void problem1(cudaStream_t stream, char* filename, double *min_dist_sq_ptr){

    Input input;
    Body *bodyArray1_dev, *bodyArray2_dev;
    BYTE *min_dist_sq_dev;

    read_input(filename, &input);

    for (int i = 0; i < input.n; i++) {
        if (input.bodyArray[i].isDevice == 1) input.bodyArray[i].m = 0;
    }

    // cudaSetDevice(0);

    cudaMalloc(&bodyArray1_dev, input.n * sizeof(Body));
    cudaMemcpy((BYTE *)bodyArray1_dev, (BYTE *)(input.bodyArray),
                            input.n * sizeof(Body), cudaMemcpyHostToDevice);

    cudaMalloc(&bodyArray2_dev, input.n * sizeof(Body));
    cudaMemcpy((BYTE *)bodyArray2_dev, (BYTE *)(input.bodyArray),
                            input.n * sizeof(Body), cudaMemcpyHostToDevice);

    double min_dist_sq_host = std::numeric_limits<double>::infinity();
    double min_dist_host;

    cudaMalloc(&min_dist_sq_dev, sizeof(double));
    cudaMemcpyAsync(min_dist_sq_dev, (BYTE *)&min_dist_sq_host,
                                    sizeof(double), cudaMemcpyHostToDevice);

    int n_block = input.n / N_THRD_PER_BLK_X + 1;
    dim3 nThreadsPerBlock(N_THRD_PER_BLK_X, N_THRD_PER_BLK_Y, 1);

    int n_batch = input.n / BATCH_SIZE;
    if(n_batch * BATCH_SIZE < input.n) n_batch += 1;

    // auto start = high_resolution_clock::now();

    for (int step = 1; step <= n_steps + 1; step++) {

        kernel_problem1<<<n_block, nThreadsPerBlock, 0, stream>>>\
                (step, n_batch, input.n, input.planetId, input.asteroidId, 
                             bodyArray1_dev, bodyArray2_dev, min_dist_sq_dev);
        
        Body *tmp = bodyArray1_dev;
        bodyArray1_dev = bodyArray2_dev;
        bodyArray2_dev = tmp;
    }


    // auto stop = high_resolution_clock::now();
    // auto duration = duration_cast<microseconds>(stop - start);
    // cout<<"problem 1 time: "<<duration.count() / 1000000. <<" sec"<<endl;

    cudaMemcpyAsync((BYTE *)min_dist_sq_ptr, min_dist_sq_dev, 
                                    sizeof(double), cudaMemcpyDeviceToHost);    

}






void problem2(cudaStream_t stream, char* filename, int *hit_time_step_ptr){

    Input input;
    Body *bodyArray1_dev, *bodyArray2_dev;
    BYTE *hit_time_step_dev;

    read_input(filename, &input);

    cudaMalloc(&bodyArray1_dev, input.n * sizeof(Body));
    cudaMemcpy((BYTE *)bodyArray1_dev, (BYTE *)(input.bodyArray),
                            input.n * sizeof(Body), cudaMemcpyHostToDevice);

    cudaMalloc(&bodyArray2_dev, input.n * sizeof(Body));
    cudaMemcpy((BYTE *)bodyArray2_dev, (BYTE *)(input.bodyArray),
                            input.n * sizeof(Body), cudaMemcpyHostToDevice);

    int  hit_time_step_host = -2;
 
    cudaMalloc(&hit_time_step_dev, sizeof(int));
    cudaMemcpyAsync(hit_time_step_dev, (BYTE *)&hit_time_step_host,
                                    sizeof(int), cudaMemcpyHostToDevice);

    int n_block = input.n / N_THRD_PER_BLK_X + 1;
    dim3 nThreadsPerBlock(N_THRD_PER_BLK_X, N_THRD_PER_BLK_Y, 1);

    int n_batch = input.n / BATCH_SIZE;
    if(n_batch * BATCH_SIZE < input.n) n_batch += 1;




    auto start = high_resolution_clock::now();

    for (int step = 1; step <= n_steps + 1; step++) {

        kernel_problem2<<<n_block, nThreadsPerBlock, 0, stream>>>\
                (step, n_batch, input.n, input.planetId, input.asteroidId, 
                             bodyArray1_dev, bodyArray2_dev, hit_time_step_dev);

        Body *tmp = bodyArray1_dev;
        bodyArray1_dev = bodyArray2_dev;
        bodyArray2_dev = tmp;

        if((step & (16 - 1)) == 0){
            cudaMemcpyAsync((BYTE *)hit_time_step_ptr, hit_time_step_dev, 
                                            sizeof(int), cudaMemcpyDeviceToHost);
            // print();
            if(*hit_time_step_ptr != -2) break;
        }

    }


    cudaDeviceSynchronize();

    cudaMemcpy((BYTE *)hit_time_step_ptr, hit_time_step_dev, 
                                    sizeof(int), cudaMemcpyDeviceToHost);

    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(stop - start);
    cout<<"problem 2 time: "<<duration.count() / 1000000. <<" sec"<<endl;

}






void problem3(cudaStream_t stream, char* filename, int hit_time_step, 
                                        int *gravity_device_id_ptr, double *missile_cost_ptr){

    Input input;
    Body *bodyArray1_dev, *bodyArray2_dev;
    // BYTE *hit_time_step_dev;

    read_input(filename, &input);

    cudaMalloc(&bodyArray1_dev, input.n * sizeof(Body));
    cudaMemcpy((BYTE *)bodyArray1_dev, (BYTE *)(input.bodyArray),
                            input.n * sizeof(Body), cudaMemcpyHostToDevice);

    cudaMalloc(&bodyArray2_dev, input.n * sizeof(Body));
    cudaMemcpy((BYTE *)bodyArray2_dev, (BYTE *)(input.bodyArray),
                            input.n * sizeof(Body), cudaMemcpyHostToDevice);


    int n_block = input.n / N_THRD_PER_BLK_X + 1;
    dim3 nThreadsPerBlock(N_THRD_PER_BLK_X, N_THRD_PER_BLK_Y, 1);

    int n_batch = input.n / BATCH_SIZE;
    if(n_batch * BATCH_SIZE < input.n) n_batch += 1;


    BYTE *missile_cost_dev;
    double missile_cost_host;
    cudaMalloc(&missile_cost_dev, sizeof(int));


    BYTE *success_dev;
    int success_host;
    cudaMalloc(&success_dev, sizeof(int));


    int gravity_device_id_min = -1;
    double missile_cost_min = std::numeric_limits<double>::infinity();

    bool success;  

    if(hit_time_step != -2){

        for(int i = 0; i < n; i++){

            if(input.Body[i].isDevice != 1) continue;
            if(input.Body[i].m[i] == 0) continue;

            gravity_device_id = i;
            success = true;

            success_host = 1;
            cudaMemcpy(success_dev, (BYTE *)&success_host
                                sizeof(int), cudaMemcpyHostToDevice);            



            for (int step = 0; step <= n_steps; step++) {

                kernel_problem3<<<n_block, nThreadsPerBlock, 0, stream>>>\
                        (step, n_batch, input.n, input.planetId, input.asteroidId, 
                        bodyArray1_dev, bodyArray2_dev, 
                        gravity_device_id, missile_cost_dev, success_dev);


                if((step & (16 - 1)) == 0){
                    // cudaMemcpyAsync((BYTE *)missile_cost_host, missile_cost_dev, 
                    //                                 sizeof(double), cudaMemcpyDeviceToHost);

                    cudaMemcpyAsync((BYTE *)success_host, success_dev, 
                                                    sizeof(int), cudaMemcpyDeviceToHost);

                    if(success_host != 1) break;
                }              
            }

            cudaDeviceSynchronize();
            cudaMemcpy((BYTE *)success_host, success_dev, 
                                            sizeof(double), cudaMemcpyDeviceToHost);
            cudaMemcpy((BYTE *)missile_cost_host, missile_cost_dev, 
                                            sizeof(double), cudaMemcpyDeviceToHost);


            if(success_host){
                if(missile_cost_host < missile_cost_min){
                    missile_cost_min = missile_cost_host;
                    gravity_device_id_min = gravity_device_id;
                }
            }
            // read_input(argv[1], n, planet, asteroid, qx, qy, qz, vx, vy, vz, m, type);
        }

    }

    if(gravity_device_id_min == -1){
        gravity_device_id = -1;
        missile_cost = 0;
    }


}





int hit_time_step;
double min_dist;
int gravity_device_id;
double missile_cost;

int main(int argc, char **argv)
{

    cudaSetDevice(0);
    cudaStream_t stream0[2];
    for (int i = 0; i < 2; ++i) cudaStreamCreate(&stream0[i]);
    
    cudaSetDevice(1);
    cudaStream_t stream1[2];
    for (int i = 0; i < 2; ++i) cudaStreamCreate(&stream1[i]);
    




    // cudaSetDevice(0);
    // double min_dist_sq;
    // problem1(stream0[0], argv[1], &min_dist_sq);
    // cudaDeviceSynchronize();
    // min_dist = sqrt(min_dist_sq);
    // printf("min_dist: %f\n", min_dist);




    // cudaSetDevice(0);
    // problem2(stream0[1], argv[1], &hit_time_step);
    // printf("hit_time_step: %d\n", hit_time_step);



    hit_time_step = 10;
    cudaSetDevice(0);
    problem3(stream0[1], argv[1], hit_time_step, &gravity_device_id, &missile_cost);


  





    return 0;
}