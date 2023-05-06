
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

__host__ __device__ 
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
    int *id_map;

    int n_dev;
};




void swapBody(Input *input, int idx1, int idx2){
    if(idx1 == idx2) return;

    Body tmpBody = input->bodyArray[idx1];
    input->bodyArray[idx1] = input->bodyArray[idx2];
    input->bodyArray[idx2] = tmpBody;

    int tmpId = input->id_map[idx1];
    input->id_map[idx1] = input->id_map[idx2];
    input->id_map[idx2] = tmpId;
}



void read_input(const char* filename, Input *input) {

    std::ifstream fin(filename);
    fin >> input->n >> input->planetId >> input->asteroidId;

    input->bodyArray = new Body[input->n];
    input->id_map = new int[input->n];
    input->n_dev = 0;

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
            input->n_dev++;
        }

        input->id_map[i] = i;
    }
}




__global__ void kernel_problem1(int step, int n, 
                        Body *bodyArray, Body *bodyArray_update, BYTE *min_dist_sq){


    int bodyId_this = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.y * blockDim.x + threadIdx.x;

    double ax = 0, ay = 0, az = 0, dx, dy, dz;
    double qx, qy, qz;
   
    if(bodyId_this < n){
        qx = bodyArray[bodyId_this].qx;
        qy = bodyArray[bodyId_this].qy;
        qz = bodyArray[bodyId_this].qz;
    }



    __shared__ WORD sm[BATCH_SIZE_WORD + N_THRD_PER_BLK_Y * 3 * 2 * N_THRD_PER_BLK_X];
    double *sm_aggregate = (double *)(sm + BATCH_SIZE_WORD);

    int n_batch = n / BATCH_SIZE;
    if(n_batch * BATCH_SIZE < n) n_batch += 1;


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

            dx = ((Body *)sm)[threadIdx.y].qx - qx;
            dy = ((Body *)sm)[threadIdx.y].qy - qy;
            dz = ((Body *)sm)[threadIdx.y].qz - qz;

            double dist3 = pow(dx * dx + dy * dy + dz * dz + eps * eps, 1.5);

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

    // double dvx = 0, dvy = 0, dvz = 0;
    // if(threadIdx.y == 0){
    //     for(int i = 1; i < blockDim.y; i++){
    //         dvx += sm_aggregate[i * (3 * blockDim.x) + 3 * threadIdx.x + 0];
    //         dvy += sm_aggregate[i * (3 * blockDim.x) + 3 * threadIdx.x + 1];
    //         dvz += sm_aggregate[i * (3 * blockDim.x) + 3 * threadIdx.x + 2];
    //     }

    //     sm_aggregate[3 * threadIdx.x + 0] += dvx;
    //     sm_aggregate[3 * threadIdx.x + 1] += dvy;
    //     sm_aggregate[3 * threadIdx.x + 2] += dvz;
    // }




    double *q_ptr_update_sm = sm_aggregate + (3 * blockDim.x + 3 * threadIdx.x);

    if((threadIdx.y < 3) && (bodyId_this < n)){
        
        double *v_ptr = (double *)&(bodyArray[bodyId_this].vx);
        double *q_ptr_update = (double *)&(bodyArray_update[bodyId_this].qx);
        double *v_ptr_update = (double *)&(bodyArray_update[bodyId_this].vx);
        
        q_ptr_update_sm[0] = qx;
        q_ptr_update_sm[1] = qy;
        q_ptr_update_sm[2] = qz;
        
        double vi = v_ptr[threadIdx.y];
        vi += sm_aggregate[3 * threadIdx.x + threadIdx.y];

        q_ptr_update_sm[threadIdx.y] += vi * dt;

        v_ptr_update[threadIdx.y] = vi;
        q_ptr_update[threadIdx.y] = q_ptr_update_sm[threadIdx.y];
        
    }

    __syncthreads();


    if((bodyId_this == 0) && (threadIdx.y == 0)){

        dx = q_ptr_update_sm[0] - q_ptr_update_sm[3 + 0];
        dy = q_ptr_update_sm[1] - q_ptr_update_sm[3 + 1];
        dz = q_ptr_update_sm[2] - q_ptr_update_sm[3 + 2];

        double min_dist_sq_local = *((double *)min_dist_sq);
        double r_sq = dx * dx + dy * dy + dz * dz;

        if(min_dist_sq_local > r_sq){
            *((double *)min_dist_sq) = r_sq;
        }  
    }
}

struct DevDistroyCkpt{
    int deviceId;
    int step;
    Body *bodyArray;
};



__global__ void kernel_problem2(int step, int n, Body *bodyArray, Body *bodyArray_update, 
                                BYTE *hit_time_step, DevDistroyCkpt *ddckptArray, int n_dev){


    if(*((int *)hit_time_step) != -2) return;
        
    int bodyId_this = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.y * blockDim.x + threadIdx.x;                          
    

    double ax = 0, ay = 0, az = 0, dx, dy, dz;
    double qx, qy, qz;
   
    if(bodyId_this < n){
        qx = bodyArray[bodyId_this].qx;
        qy = bodyArray[bodyId_this].qy;
        qz = bodyArray[bodyId_this].qz;
    }


    // check missile hit device.
    if(bodyId_this < n){

        double travel_dist = step * dt * missile_speed;

        for(int i = 0; i < n_dev; i++){

            if(ddckptArray[i].step > 0) continue;

            int deviceId = ddckptArray[i].deviceId;

            dx = bodyArray[0].qx - bodyArray[deviceId].qx;
            dy = bodyArray[0].qy - bodyArray[deviceId].qy;
            dz = bodyArray[0].qz - bodyArray[deviceId].qz;

            if (dx * dx + dy * dy + dz * dz < travel_dist * travel_dist){

                // if(bodyId_this == 0 && threadIdx.y == 0){
                //     printf("deviceId: %d, step: %d\n", deviceId, step);
                // }

                ddckptArray[i].step = step;

                double vx = bodyArray[bodyId_this].vx;
                double vy = bodyArray[bodyId_this].vy;
                double vz = bodyArray[bodyId_this].vz;
                
                double m;
                if(deviceId == bodyId_this){m = 0.;}
                else{m = bodyArray[bodyId_this].m;}

                long long isDevice = bodyArray[bodyId_this].isDevice;
                
                ddckptArray[i].bodyArray[bodyId_this].qx = qx;
                ddckptArray[i].bodyArray[bodyId_this].qx = qx;
                ddckptArray[i].bodyArray[bodyId_this].qx = qx;
                
                ddckptArray[i].bodyArray[bodyId_this].vx = vx;
                ddckptArray[i].bodyArray[bodyId_this].vx = vx;
                ddckptArray[i].bodyArray[bodyId_this].vx = vx;

                ddckptArray[i].bodyArray[bodyId_this].m = m;
                ddckptArray[i].bodyArray[bodyId_this].isDevice = isDevice;
            }
        }
    }
    



    __shared__ WORD sm[BATCH_SIZE_WORD + N_THRD_PER_BLK_Y * 3 * 2 * N_THRD_PER_BLK_X];
    double *sm_aggregate = (double *)(sm + BATCH_SIZE_WORD);

    int n_batch = n / BATCH_SIZE;
    if(n_batch * BATCH_SIZE < n) n_batch += 1;

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

            double dist3 = pow(dx * dx + dy * dy + dz * dz + eps * eps, 1.5);

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

    double *q_ptr_update_sm = sm_aggregate + (3 * blockDim.x + 3 * threadIdx.x);

    if((threadIdx.y < 3) && (bodyId_this < n)){
        
        double *v_ptr = (double *)&(bodyArray[bodyId_this].vx);
        double *q_ptr_update = (double *)&(bodyArray_update[bodyId_this].qx);
        double *v_ptr_update = (double *)&(bodyArray_update[bodyId_this].vx);
   
        q_ptr_update_sm[0] = qx;
        q_ptr_update_sm[1] = qy;
        q_ptr_update_sm[2] = qz;
        
        double vi = v_ptr[threadIdx.y];
        vi += sm_aggregate[3 * threadIdx.x + threadIdx.y];

        q_ptr_update_sm[threadIdx.y] += vi * dt;

        v_ptr_update[threadIdx.y] = vi;
        q_ptr_update[threadIdx.y] = q_ptr_update_sm[threadIdx.y];
        
    }

    __syncthreads();

    if((bodyId_this == 0) && (threadIdx.y == 0)){

        dx = q_ptr_update_sm[0] - q_ptr_update_sm[3 + 0];
        dy = q_ptr_update_sm[1] - q_ptr_update_sm[3 + 1];
        dz = q_ptr_update_sm[2] - q_ptr_update_sm[3 + 2];

        if (dx * dx + dy * dy + dz * dz < planet_radius * planet_radius) {
            
            *((int *)hit_time_step) = step; 
        }
    }

}




__global__ void kernel_problem3(int step, int n,
                        Body *bodyArray, Body *bodyArray_update, BYTE *success){

    if(*((int *)success) == 0) return;


    int bodyId_this = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.y * blockDim.x + threadIdx.x;

    double ax = 0, ay = 0, az = 0, dx, dy, dz;
    double qx, qy, qz;
   
    if(bodyId_this < n){
        qx = bodyArray[bodyId_this].qx;
        qy = bodyArray[bodyId_this].qy;
        qz = bodyArray[bodyId_this].qz;
    }

    // check asteroid hit planet.
    if((bodyId_this == 1) && (threadIdx.y == 0)){

        dx = bodyArray[0].qx - qx;
        dy = bodyArray[0].qy - qy;
        dz = bodyArray[0].qz - qz;

        if (dx * dx + dy * dy + dz * dz < planet_radius * planet_radius) {
            *((int *)success) = 0;
        }
    }



    __shared__ WORD sm[BATCH_SIZE_WORD + N_THRD_PER_BLK_Y * 3 * 2 * N_THRD_PER_BLK_X];
    double *sm_aggregate = (double *)(sm + BATCH_SIZE_WORD);

    int n_batch = n / BATCH_SIZE;
    if(n_batch * BATCH_SIZE < n) n_batch += 1;

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

            double dist3 = pow(dx * dx + dy * dy + dz * dz + eps * eps, 1.5);

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


    double *q_ptr_update_sm = sm_aggregate + (3 * blockDim.x + 3 * threadIdx.x);

    if((threadIdx.y < 3) && (bodyId_this < n)){
        
        double *v_ptr = (double *)&(bodyArray[bodyId_this].vx);
        double *q_ptr_update = (double *)&(bodyArray_update[bodyId_this].qx);
        double *v_ptr_update = (double *)&(bodyArray_update[bodyId_this].vx);
        
        q_ptr_update_sm[0] = qx;
        q_ptr_update_sm[1] = qy;
        q_ptr_update_sm[2] = qz;
        
        double vi = v_ptr[threadIdx.y];
        vi += sm_aggregate[3 * threadIdx.x + threadIdx.y];

        q_ptr_update_sm[threadIdx.y] += vi * dt;

        v_ptr_update[threadIdx.y] = vi;
        q_ptr_update[threadIdx.y] = q_ptr_update_sm[threadIdx.y];
        
    }

}



class KCB1{

    public:

    KCB1(int gpuId, char* filename){

        this->gpuId = gpuId;
        cudaSetDevice(gpuId);
        

        init_commom(filename);
        init();
        cpy_h2d_setup_common();
        cpy_h2d_setup();
    }

    void init_commom(char* filename){
        
        input = new Input();
        read_input(filename, input);
        
        step = 1;

        cudaStreamCreate(&stream);
    
        n_block = input->n / N_THRD_PER_BLK_X + 1;
        nThreadsPerBlock = dim3(N_THRD_PER_BLK_X, N_THRD_PER_BLK_Y, 1);

        cudaMalloc(&bodyArray1_dev, input->n * sizeof(Body));
        cudaMalloc(&bodyArray2_dev, input->n * sizeof(Body));
    }


    // problem specific
    void init(){

        swapBody(input, input->planetId, 0);
        swapBody(input, input->asteroidId, 1);

        cudaMalloc(&min_dist_sq_dev, sizeof(double));
        
        double dx = input->bodyArray[0].qx - input->bodyArray[1].qx;
        double dy = input->bodyArray[0].qy - input->bodyArray[1].qy;
        double dz = input->bodyArray[0].qz - input->bodyArray[1].qz;

        min_dist_sq_host = dx * dx + dy * dy + dz * dz; 
        
        for (int i = 0; i < input->n; i++) {
            if (input->bodyArray[i].isDevice == 1) input->bodyArray[i].m = 0;
        }        
    }

    void cpy_h2d_setup_common(){
        cudaMemcpy((BYTE *)bodyArray1_dev, (BYTE *)(input->bodyArray),
                                input->n * sizeof(Body), cudaMemcpyHostToDevice);
        cudaMemcpy((BYTE *)bodyArray2_dev, (BYTE *)(input->bodyArray),
                                input->n * sizeof(Body), cudaMemcpyHostToDevice);
    }

    // problem specific
    void cpy_h2d_setup(){
        cudaMemcpy(min_dist_sq_dev, (BYTE *)&min_dist_sq_host,
                                        sizeof(double), cudaMemcpyHostToDevice);
    }



    // problem specific
    void cpy_d2h_return(){
        cudaSetDevice(gpuId);
        cudaMemcpy((BYTE *)&min_dist_sq_host, min_dist_sq_dev, 
                                        sizeof(double), cudaMemcpyDeviceToHost); 
        min_dist_host = sqrt(min_dist_sq_host);                       
    }

    bool done(){
        return (step - 1 >= n_steps);
    }

    // problem specific
    void one_step(){

        if(done()) return;
        
        cudaSetDevice(gpuId);

        kernel_problem1<<<n_block, nThreadsPerBlock, 0, stream>>>\
                (step, input->n, bodyArray1_dev, bodyArray2_dev, min_dist_sq_dev);
              
        step++;

        Body *tmp = bodyArray1_dev;
        bodyArray1_dev = bodyArray2_dev;
        bodyArray2_dev = tmp;
    }

    int gpuId;
    int n_block;
    dim3 nThreadsPerBlock;

    int step;
    Input *input;
    Body *bodyArray1_dev, *bodyArray2_dev;
    cudaStream_t stream;

    // problem specific
    BYTE *min_dist_sq_dev;
    double min_dist_sq_host;
    double min_dist_host;
};





class KCB2{

    public:

    KCB2(int gpuId, char* filename){

        this->gpuId = gpuId;

        cudaSetDevice(gpuId);

        init_commom(filename);
        init();
        cpy_h2d_setup_common();
        cpy_h2d_setup();        
    }

    void init_commom(char* filename){
        input = new Input();
        read_input(filename, input);
        step = 1;

        cudaStreamCreate(&stream);

        n_block = input->n / N_THRD_PER_BLK_X + 1;
        nThreadsPerBlock = dim3(N_THRD_PER_BLK_X, N_THRD_PER_BLK_Y, 1);

        cudaMalloc(&bodyArray1_dev, input->n * sizeof(Body));
        cudaMalloc(&bodyArray2_dev, input->n * sizeof(Body));
    }


    // problem specific
    void init(){   

        swapBody(input, input->planetId, 0);
        swapBody(input, input->asteroidId, 1);

        init_ddckptArray();

        hit_time_step_host = -2;

        double dx = input->bodyArray[0].qx - input->bodyArray[1].qx;
        double dy = input->bodyArray[0].qy - input->bodyArray[1].qy;
        double dz = input->bodyArray[0].qz - input->bodyArray[1].qz;

        if (dx * dx + dy * dy + dz * dz < planet_radius * planet_radius) {
            hit_time_step_host = 0; 
        }

        cudaMalloc(&hit_time_step_dev, sizeof(int));           
    }


    void init_ddckptArray(){
        
        ddckptArray_host = new DevDistroyCkpt[input->n_dev];

        int *deviceIdArray = new int[input->n_dev];
        int idx = 0;
        for (int i = 0; i < input->n; i++) {
            if (input->bodyArray[i].isDevice == 1){
                deviceIdArray[idx++] = i;
            }
        }
        
        for(int i = 0; i < input->n_dev; i++){
            ddckptArray_host[i].deviceId = deviceIdArray[i];
            ddckptArray_host[i].step = -1;
            cudaMalloc(&(ddckptArray_host[i].bodyArray), input->n * sizeof(Body)); 
        }

        cudaMalloc(&ddckptArray_dev, input->n_dev * sizeof(DevDistroyCkpt)); 

        cudaMemcpy((BYTE *)ddckptArray_dev, (BYTE *)ddckptArray_host,
                        input->n_dev * sizeof(DevDistroyCkpt), cudaMemcpyHostToDevice);        

    }

    void cpy_h2d_setup_common(){
        cudaMemcpy((BYTE *)bodyArray1_dev, (BYTE *)(input->bodyArray),
                                input->n * sizeof(Body), cudaMemcpyHostToDevice);
        cudaMemcpy((BYTE *)bodyArray2_dev, (BYTE *)(input->bodyArray),
                                input->n * sizeof(Body), cudaMemcpyHostToDevice);
    }

    // problem specific
    void cpy_h2d_setup(){
   
        cudaMemcpy(hit_time_step_dev, (BYTE *)&hit_time_step_host,
                                        sizeof(int), cudaMemcpyHostToDevice);                                        
    }


    bool done(){
        return ((hit_time_step_host != -2) || (step - 1 >= n_steps));
    }   

    // problem specific
    void cpy_d2h_return(){
        cudaSetDevice(gpuId);
        cudaMemcpy((BYTE *)&hit_time_step_host, hit_time_step_dev, 
                                        sizeof(int), cudaMemcpyDeviceToHost);          

        cudaMemcpy((BYTE *)ddckptArray_host, (BYTE *)ddckptArray_dev, 
                        input->n_dev * sizeof(DevDistroyCkpt), cudaMemcpyDeviceToHost);
    }

    // problem specific
    void cpy_async_d2h_return(){
        cudaSetDevice(gpuId);

        cudaMemcpyAsync((BYTE *)&hit_time_step_host, hit_time_step_dev, 
                                        sizeof(int), cudaMemcpyDeviceToHost);    

        cudaMemcpyAsync((BYTE *)ddckptArray_host, (BYTE *)ddckptArray_dev, 
                        input->n_dev * sizeof(DevDistroyCkpt), cudaMemcpyDeviceToHost);
    }


    // problem specific
    void one_step(){
        
        if(done()) return;
        if((step & (16 - 1)) == 0) cpy_async_d2h_return();

        cudaSetDevice(gpuId);

        kernel_problem2<<<n_block, nThreadsPerBlock, 0, stream>>>\
                (step, input->n, bodyArray1_dev, bodyArray2_dev,
                     hit_time_step_dev, ddckptArray_dev, input->n_dev);

        step++;

        Body *tmp = bodyArray1_dev;
        bodyArray1_dev = bodyArray2_dev;
        bodyArray2_dev = tmp;
    }


    void free(){
        cudaSetDevice(gpuId);
        cudaFree(bodyArray1_dev);
        cudaFree(bodyArray2_dev);
        cudaFree(hit_time_step_dev);
    }

    int gpuId;
    int n_block;
    dim3 nThreadsPerBlock;

    int step;
    Input *input;
    Body *bodyArray1_dev, *bodyArray2_dev;
    cudaStream_t stream;

    // problem specific
    BYTE *hit_time_step_dev;
    int  hit_time_step_host = -2;
    DevDistroyCkpt *ddckptArray_dev, *ddckptArray_host;
};




__global__ void kernel_bodyArray_cpy(int n, Body *bodyArray_src, Body *bodyArray_dst){

    int tid = threadIdx.x;
    int nTrd = blockDim.x;
    int n_word = n * BODY_SIZE_WORD;

    int n_batch = n_word / nTrd;
    
    if(n_batch * nTrd < n_word) n_batch++;

    for(int i = 0; i < n_batch; i++){

        if(i < n_word){
            bodyArray_dst[i * nTrd + tid] = bodyArray_src[i * nTrd + tid];
        }
    }
    
}


class KCB_2_3{

    public:

    KCB_2_3(int gpuId, char* filename){

        kcb2 = new KCB2(gpuId, filename);
        kcb3 = new KCB3(gpuId, kcb2);
    }

    void one_step(){
        kcb2->one_step();
        kcb3->check_new_job();
        kcb3->one_step();
    }


    int gpuId;
    KCB2 *kcb2;
    KCB3 *kcb3;
}


class KCB3{

    public:

    KCB3(int gpuId, KCB2 *kcb2){

        this->ddckptArray_dev = kcb2->ddckptArray_dev;
        this->ddckptArray_host = kcb2->ddckptArray_host;
        this->n_dev = kcb2->input->n_dev;
        this->input = kcb2->input;
        this->kcb2 = kcb2;

        cudaSetDevice(gpuId);

        init_commom();
        init();
        // cpy_h2d_setup_common();
        // cpy_h2d_setup();        
    }

    void init_commom(){

        for (int i = 0; i < n_stream; ++i) cudaStreamCreate(&stream[i]);

        n_block = input->n / N_THRD_PER_BLK_X + 1;
        nThreadsPerBlock = dim3(N_THRD_PER_BLK_X, N_THRD_PER_BLK_Y, 1);
    }


    // problem specific
    void init(){   

        stepArray = new int[n_dev];
        ddstepArray = new int[n_dev];
        for(int i = 0; i < n_dev; i++){
            stepArray[i] = -1;
        }
        
        // for(int i = 0; i < n_dev; i++){
        //     cudaMemcpy((BYTE *)(stepArray + i), ((BYTE *)(ddckptArray_dev)) + \
        //          4 + i * sizeof(DevDistroyCkpt), sizeof(int), cudaMemcpyDeviceToHost);   
        //     ddstepArray[i] = stepArray[i];
        // }



        bodyArray1_dev_array = new Body*[n_dev];
        bodyArray2_dev_array = new Body*[n_dev];
        for(int i = 0; i < n_dev; i++){
            bodyArray1_dev_array[i] = ddckptArray_host[i].bodyArray;
            cudaMalloc(&(bodyArray2_dev_array[i]), input->n * sizeof(Body));
        }       



        success_dev_array = new BYTE*[n_dev];
        success_host_array = new int[n_dev];
        for(int i = 0; i < n_dev; i++){
            cudaMalloc(&(success_dev_array[i]), sizeof(int));
            success_host_array[i] = 1;
        }        



        // step_global = 0;
        // for(int i = 0; i < 2; i++) jobIdArray[i] = -1;
        // for(int i = 0; i < min(2, n_dev); i++) jobIdArray[i] = i;

        // JobId_next = -1;
        // for(int i = 0; i < 2; i++){
        //     if(jobIdArray[i] != -1) JobId_next = jobIdArray[i] + 1;
        // }
        // if(JobId_next >= n_dev) JobId_next = -1;
    }


    void cpy_h2d_setup_common(){

        // for(int i = 0; i < n_dev; i++){

        //     kernel_bodyArray_cpy<<<1, 512, 0, stream[i % 2]>>>\
        //                     (input->n, bodyArray1_dev_array[i], bodyArray2_dev_array[i]);
        // }

        // cudaDeviceSynchronize();    
    }

    // problem specific
    void cpy_h2d_setup(int jobId, int streamId){

        kernel_bodyArray_cpy<<<1, 512, 0, stream[streamId]>>>\
                        (input->n, bodyArray1_dev_array[jobId], bodyArray2_dev_array[jobId]);
                        
        cudaMemcpy(success_dev_array[jobId], (BYTE *)&(success_host_array[jobId]),
                                        sizeof(int), cudaMemcpyHostToDevice); 

        // cudaStreamSynchronize();
        cudaStreamSynchronize(stream[streamId]);
    }


    void check_new_job(){
        
        for(int i = 0; i < n_dev; i++){
            if(stepArray[i] != -1) continue;
            if(kcb2->ddckptArray_host[i].step == -1) continue;

            stepArray[i] = kcb2->ddckptArray_host[i].step;
            cpy_h2d_setup(i, streamId_next);
            streamId_next = (streamId_next + 1) % n_stream;
        }
    }
    


    void cpy_d2h_return(int i){
        cudaSetDevice(gpuId);
        cudaMemcpy((BYTE *)&(success_host_array[i]), success_dev_array[i], 
                                        sizeof(int), cudaMemcpyDeviceToHost);                  
    }


    void cpy_async_d2h_return(int i){
        cudaSetDevice(gpuId);
        cudaMemcpyAsync((BYTE *)&(success_host_array[i]), success_dev_array[i], 
                                        sizeof(int), cudaMemcpyDeviceToHost);                   
    }


    bool done(){
        bool done = true;
        for(int i = 0; i < n_dev; i ++) done = done && done_job(i);
        return done;
    }


    bool done_job(int jobId){
        return ((stepArray[JobId] - 1 >= n_steps) || (success_host_array[jobId] == 0));
    }


    void one_step(){
        
        for(int i = 0; i < n_dev; i ++){
            
            if(stepArray[i] == -1) continue;
            if(done(i)) continue;

            cudaSetDevice(gpuId);

            kernel_problem3<<<n_block, nThreadsPerBlock, 0, stream[streamId_next]>>>\
                    (stepArray[i], input->n, bodyArray1_dev_array[i], bodyArray2_dev_array[i],
                        success_dev_array[i]);

            streamId_next = (streamId_next + 1) % n_stream;
            stepArray[i]++;

            Body *tmp = bodyArray1_dev_array[i];
            bodyArray1_dev_array[i] = bodyArray2_dev_array[i];
            bodyArray2_dev_array[i] = tmp;

            if(((stepArray[i]) & (16 - 1)) == 0) cpy_async_d2h_return(i);
        }
    }    


    void process_return(){

        int min_step = n_steps + 1;
        gravity_device_id = -1;

        for(int i = 0; i < n_dev; i++){

            if(success_host_array[i] == 0)continue;

            if(ddstepArray[i] < min_step) {
                min_step = ddstepArray[i];
                gravity_device_id = ddckptArray_host[i].deviceId;
            }
        }
        if(gravity_device_id == -1) {missile_cost = 0.;}
        else {missile_cost = get_missile_cost(min_step * dt);}
    }




    int gpuId;
    int n_block;
    dim3 nThreadsPerBlock;
    int jobIdArray[2];
    int JobId_next;

    int *stepArray;
    int *ddstepArray;
    int step_global;
    Input *input;
    Body **bodyArray1_dev_array, **bodyArray2_dev_array;

    int streamId_next = 0;
    cudaStream_t stream[4];
    int n_stream = 4;

    // problem specific
    KCB2 *kcb2;
    BYTE **success_dev_array;
    int *success_host_array;
    DevDistroyCkpt *ddckptArray_dev;
    DevDistroyCkpt *ddckptArray_host;
    int n_dev;

    double missile_cost;
    int gravity_device_id;
};





int hit_time_step;
double min_dist;
int gravity_device_id;
double missile_cost;

int main(int argc, char **argv)
{

    cudaSetDevice(0);
    cudaStream_t stream0[2];
    for (int i = 0; i < 2; ++i) cudaStreamCreate(&stream0[i]);
    


    auto start = high_resolution_clock::now();
    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(stop - start);

    // start = high_resolution_clock::now();

    // -----------------------------------------------------------

    KCB1 kcb1(1, argv[1]);

    while(!kcb1.done()){
        kcb1.one_step();
    }


    // -----------------------------------------------------------

    KCB2 kcb2(0, argv[1]);

    while(!kcb2.done()){
        kcb2.one_step();
    }

    // -----------------------------------------------------------

    KCB3 kcb3(0, &kcb2);

    while(!kcb3.all_job_done()){
        kcb3.one_step();
    }



    // -----------------------------------------------------------

    cudaSetDevice(1);

    cudaDeviceSynchronize();
    kcb1.cpy_d2h_return();
    printf("min_dist: %f\n", kcb1.min_dist_host);

    cudaSetDevice(0);

    if(kcb2.hit_time_step_host != -2){
        hit_time_step = kcb2.hit_time_step_host;
    }else{
        cudaDeviceSynchronize();
        kcb2.cpy_d2h_return();
        hit_time_step = kcb2.hit_time_step_host;
    }
    // kcb2.free();
    printf("hit_time_step: %d\n", hit_time_step);


    cudaDeviceSynchronize();
    for(int i=0;i<kcb3.n_dev;i++){
        kcb3.cpy_d2h_return(i);
    }
    kcb3.process_return();
    gravity_device_id = kcb3.gravity_device_id;
    missile_cost = kcb3.missile_cost;
    printf("gravity_device_id: %d\n", gravity_device_id);
    printf("missile_cost: %f\n", missile_cost);


    // -----------------------------------------------------------

    
    // stop = high_resolution_clock::now();
    // duration = duration_cast<microseconds>(stop - start);
    // cout<<"problem 1 time: "<<duration.count() / 1000000. <<" sec"<<endl;



    // cudaSetDevice(0);
    // problem2(stream0[1], argv[1], &hit_time_step);
    // printf("hit_time_step: %d\n", hit_time_step);






    // hit_time_step = 10;
    // cudaSetDevice(0);
    // auto start = high_resolution_clock::now();
    // problem3(stream0[1], argv[1], hit_time_step, &gravity_device_id, &missile_cost);
    // auto stop = high_resolution_clock::now();
    // auto duration = duration_cast<microseconds>(stop - start);
    // cout<<"problem 3 time: "<<duration.count() / 1000000. <<" sec"<<endl;
    // printf("gravity_device_id: %d, missile_cost: %f\n", gravity_device_id, missile_cost);
  





    return 0;
}