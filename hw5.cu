
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





#define N_BLK 1
#define N_THRD_PER_BLK 32




__global__ void kernel1()
{

    int gtid = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;
    double a = 1.;

    for(int i =0;i<100000000;i++){
        a *= 1.000000001;
    }

    if(gtid==0){
        printf("kernel1: %f\n", a);
    }

}


__global__ void kernel2()
{

    int gtid = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;
    double a = 4.;

    for(int i =0;i<100000000;i++){
        a *= 1.000000001;
    }

    if(gtid==0){
        printf("kernel2: %f\n", a);
    }
     
    
}


int main(int argc, char **argv)
{

    cudaStream_t stream[2];
    for (int i = 0; i < 2; ++i){
        cudaStreamCreate(&stream[i]);
    }


    auto start = high_resolution_clock::now();
    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(stop - start);


    // start = high_resolution_clock::now();

    // kernel1<<<N_BLK, N_THRD_PER_BLK, 0, 0>>>();
    // kernel2<<<N_BLK, N_THRD_PER_BLK, 0, 0>>>();

    // cudaDeviceSynchronize();   

    // stop = high_resolution_clock::now();
    // duration = duration_cast<microseconds>(stop - start);
    // cout<<"no parallel, dt: "<<duration.count() / 1000000. <<" sec"<<endl;



    start = high_resolution_clock::now();

    kernel1<<<N_BLK, N_THRD_PER_BLK, 0, stream[0]>>>();
    kernel2<<<N_BLK, N_THRD_PER_BLK, 0, stream[1]>>>();

    cudaDeviceSynchronize();   

    stop = high_resolution_clock::now();
    duration = duration_cast<microseconds>(stop - start);
    cout<<"with parallel, dt: "<<duration.count() / 1000000. <<" sec"<<endl;



    // for (int i = 0; i < 2; ++i){
    //     cudaStreamDestroy(stream[i]);
    // }
    
    return 0;
}