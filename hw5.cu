
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


#define QM_SIZE_BYTE 32
#define QM_SIZE_WORD 8
#define QM_SIZE_DOUBLE 4

#define BATCH_SIZE (N_THRD_PER_BLK_Y)
#define BATCH_QM_SIZE_DOUBLE (BATCH_SIZE * QM_SIZE_DOUBLE)
#define BATCH_QM_SIZE_WORD (BATCH_SIZE * QM_SIZE_WORD)

#define FLAG_SIZE_WORD 1
#define BATCH_FLAG_SIZE_WORD (BATCH_SIZE * FLAG_SIZE_WORD)

#define BATCH_SIZE_WORD (BATCH_FLAG_SIZE_WORD + BATCH_QM_SIZE_WORD)

// need to make sure that this is int and can divide BATCH_SIZE.
#define N_QM_COPY_PER_PASS (N_THRD_PER_BLK * 4 / QM_SIZE_BYTE) // 16.


typedef unsigned int WORD;
typedef unsigned char BYTE;

// struct Body{
    
//     double qx, qy, qz, vx, vy, vz, m;
//     long long isDevice;
    
// };


struct QM{
    double qx, qy, qz, m;
};

struct Vel{
    double vx, vy, vz;
};

struct Input{
    int n;
    int planetId;
    int asteroidId;
    QM *qmArray;
    Vel *velArray;
    int *flagArray;
};


// struct Input{
//     int n;
//     int planetId;
//     int asteroidId;
//     Body *bodyArray;
// };





void read_input(const char* filename, Input *input) {

    std::ifstream fin(filename);
    fin >> input->n >> input->planetId >> input->asteroidId;

    input->qmArray = new QM[input->n];
    input->velArray = new Vel[input->n];
    input->flagArray = new int[input->n];

    string type;

    for (int i = 0; i < input->n; i++) {
        fin >> input->qmArray[i].qx 
            >> input->qmArray[i].qy
            >> input->qmArray[i].qz 
            >> input->velArray[i].vx 
            >> input->velArray[i].vy 
            >> input->velArray[i].vz 
            >> input->qmArray[i].m 
            >> type;
        
        if (type != "device"){
            input->flagArray[i] = 0;
        }
        else{
            input->flagArray[i] = 1;
        }
    }
}



__global__ void kernel_problem1(int step, int n, int planetId, int asteroidId,
                                    QM *qmArray, QM *qmArray_update, 
                                    Vel *velArray, Vel *velArray_update, 
                                    int* flagArray, BYTE *min_dist_sq){


    int bodyId_this = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.y * blockDim.x + threadIdx.x;

    double ax = 0, ay = 0, az = 0, dx, dy, dz, m;
    double qx, qy, qz;

    
    // clock_t start_time = clock(); 
    // clock_t stop_time = clock();
    // int runtime = (int)(stop_time - start_time);

   
    if(bodyId_this < n){
        qx = qmArray[bodyId_this].qx;
        qy = qmArray[bodyId_this].qy;
        qz = qmArray[bodyId_this].qz;
    }

    // update min_dist.
    if((bodyId_this == planetId) && (threadIdx.y == 0)){

        dx = qx - qmArray[asteroidId].qx;
        dy = qy - qmArray[asteroidId].qy;
        dz = qz - qmArray[asteroidId].qz;

        *((double *)min_dist_sq) = min(*((double *)min_dist_sq), 
                                             dx * dx + dy * dy + dz * dz);  
    }


    __shared__ WORD sm[BATCH_SIZE_WORD + N_THRD_PER_BLK_Y * 3 * 2 * N_THRD_PER_BLK_X];
    
    double *sm_aggregate = (double *)(sm + BATCH_SIZE_WORD);
    QM *qmArray_sm = (QM *)sm;
    int *flagArray_sm = (int *)(sm + BATCH_QM_SIZE_WORD);
    double qmArray_local[4 * BATCH_SIZE];

    int n_batch = n / BATCH_SIZE;
    if(n_batch * BATCH_SIZE < n) n_batch += 1;

    #pragma unroll
    for(int batchId = 0; batchId < n_batch; batchId++){
        
        // for(int i = 0; i < BATCH_SIZE; i += N_QM_COPY_PER_PASS){

        //     int global_offset = batchId * BATCH_QM_SIZE_WORD;
        //     int local_offset = i * QM_SIZE_WORD + tid;
        //     int idx = global_offset + local_offset;

        //     if(idx < n * QM_SIZE_WORD){
        //         sm[local_offset] = ((WORD *)qmArray)[idx];
        //     }
        // }

        int global_offset = batchId * BATCH_QM_SIZE_DOUBLE;

        #pragma unroll
        for(int i = 0; i < 4 * BATCH_SIZE; i++){    
            if(global_offset + i < n * QM_SIZE_DOUBLE){
                qmArray_local[i] = ((double *)qmArray)[global_offset + i];
            }
        }

        if(threadIdx.x == 0){

            int global_offset = batchId * BATCH_SIZE;
            int local_offset = threadIdx.y;
            int idx = global_offset + local_offset;

            if(idx < n){
                flagArray_sm[local_offset] = flagArray[idx];
            }            
        }


        __syncthreads();


        int bodyId_other = batchId * BATCH_SIZE + threadIdx.y;
        
        if ((bodyId_other != bodyId_this) && (bodyId_other < n)){
            
            double mj = qmArray_local[4 * threadIdx.y + 3]; // m
     
            if (flagArray_sm[threadIdx.y] == 1) {
                mj = gravity_device_mass(mj, step * dt);
            }

            dx = qmArray_local[4 * threadIdx.y + 0] - qx;
            dy = qmArray_local[4 * threadIdx.y + 1] - qy;
            dz = qmArray_local[4 * threadIdx.y + 2] - qz;

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
        
        
        double *v_ptr = (double *)(velArray + bodyId_this);
        double *q_ptr_update = (double *)(qmArray_update + bodyId_this);
        double *v_ptr_update = (double *)(velArray_update + bodyId_this);
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





int main(int argc, char **argv)
{
    Input input;
    QM *qmArray1_dev, *qmArray2_dev;
    Vel *velArray1_dev, *velArray2_dev;
    int *flagArray_dev;
    BYTE *min_dist_sq_dev;


    read_input(argv[1], &input);

    for (int i = 0; i < input.n; i++) {
        if (input.flagArray[i] == 1) input.qmArray[i].m = 0;
    }

    cudaSetDevice(0);

    cudaMalloc(&qmArray1_dev, input.n * sizeof(QM));
    cudaMemcpy((BYTE *)qmArray1_dev, (BYTE *)(input.qmArray),
                            input.n * sizeof(QM), cudaMemcpyHostToDevice);

    cudaMalloc(&qmArray2_dev, input.n * sizeof(QM));
    cudaMemcpy((BYTE *)qmArray2_dev, (BYTE *)(input.qmArray),
                            input.n * sizeof(QM), cudaMemcpyHostToDevice);

    cudaMalloc(&velArray1_dev, input.n * sizeof(Vel));
    cudaMemcpy((BYTE *)velArray1_dev, (BYTE *)(input.velArray),
                            input.n * sizeof(Vel), cudaMemcpyHostToDevice);

    cudaMalloc(&velArray2_dev, input.n * sizeof(Vel));
    cudaMemcpy((BYTE *)velArray2_dev, (BYTE *)(input.velArray),
                            input.n * sizeof(Vel), cudaMemcpyHostToDevice);

    cudaMalloc(&flagArray_dev, input.n * sizeof(int));
    cudaMemcpy((BYTE *)flagArray_dev, (BYTE *)(input.flagArray),
                            input.n * sizeof(int), cudaMemcpyHostToDevice);



    double min_dist_sq_host = std::numeric_limits<double>::infinity();

    cudaMalloc(&min_dist_sq_dev, sizeof(double));
    cudaMemcpy(min_dist_sq_dev, (BYTE *)&min_dist_sq_host,
                                    sizeof(double), cudaMemcpyHostToDevice);

    int n_block = input.n / N_THRD_PER_BLK_X + 1;
    dim3 nThreadsPerBlock(N_THRD_PER_BLK_X, N_THRD_PER_BLK_Y, 1);



    auto start = high_resolution_clock::now();

    for (int step = 1; step <= n_steps + 1; step++) {

        kernel_problem1<<<n_block, nThreadsPerBlock>>>(step, input.n, input.planetId, 
                                    input.asteroidId, qmArray1_dev, qmArray2_dev, velArray1_dev, 
                                    velArray2_dev, flagArray_dev, min_dist_sq_dev);
        
        QM *qmTmp = qmArray1_dev;
        qmArray1_dev = qmArray2_dev;
        qmArray2_dev = qmTmp;

        Vel *velTmp = velArray1_dev;
        velArray1_dev = velArray2_dev;
        velArray2_dev = velTmp;
    }

    cudaDeviceSynchronize();

    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(stop - start);
    cout<<"problem 1 time: "<<duration.count() / 1000000. <<" sec"<<endl;



    cudaMemcpy((BYTE *)&min_dist_sq_host, min_dist_sq_dev, 
                                    sizeof(double), cudaMemcpyDeviceToHost);    

    printf("min_dist_host: %f\n", sqrt(min_dist_sq_host));
    


    return 0;
}