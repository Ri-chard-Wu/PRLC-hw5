
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





#define N_BLK 1000
#define N_THRD_PER_BLK 512




__global__ void kernel()
{

    int gtid = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;
    double a = 1.;

    for(int i =0;i<10000000;i++){
        a *= 1.000000001;
    }

    // printf("gtid: %d\n", gtid);
    if(gtid==0){
        printf("%f\n", a);
    }
    
}




int main(int argc, char **argv)
{
    // int size = 1000;
    // unsigned char *devSrc0, *devSrc1, *hostSrc0, *hostSrc1;

    // cudaSetDevice(0);
    // // cudaMalloc(&devSrc0, size);
    // // cudaMemcpy(devSrc0, hostSrc0, size, cudaMemcpyHostToDevice);
    // kernel<<<N_BLK, N_THRD_PER_BLK>>>();

    // cudaSetDevice(1);
    // // cudaMalloc(&devSrc1, size);
    // // cudaMemcpy(devSrc0, hostSrc1, size, cudaMemcpyHostToDevice);
    // kernel<<<N_BLK, N_THRD_PER_BLK>>>();   
    // cudaDeviceSynchronize();

    // // cudaMalloc(blockHeaderDev, BLK_HDR_SIZE);
    // // cudaMemcpy(*blockHeaderDev, (unsigned char*)block, BLK_HDR_SIZE, cudaMemcpyHostToDevice);

    // // cudaMalloc(nonceValidDev, sizeof(int));
    // // cudaMemset(*nonceValidDev, 0, sizeof(int));


    auto start = high_resolution_clock::now();

    cudaSetDevice(0);
    kernel<<<N_BLK, N_THRD_PER_BLK>>>();

    cudaSetDevice(1);
    kernel<<<N_BLK, N_THRD_PER_BLK>>>();  

    cudaDeviceSynchronize();   
   
    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(stop - start);
    cout<<"time: "<<duration.count()<<" us"<<endl;




    return 0;
}