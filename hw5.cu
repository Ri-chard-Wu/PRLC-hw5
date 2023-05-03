
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





#define N_BLK 512
#define N_THRD_PER_BLK 32




__global__ void kernel(int *src, int incre)
{

    int gtid = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;
    
    if(gtid == 0){
        printf("\n----------------------\n");
    }
    
    src[gtid] += incre + gtid;

    if(tid == 0 && blockIdx.x < 10){
        printf("src[%d]: %d, incre: %d\n", gtid, src[gtid], incre);
    }    
    
}




int main(int argc, char **argv)
{
    int size = N_BLK * N_THRD_PER_BLK;
    int *devSrc0, *hostSrc0;
    hostSrc0 = new int[size];
    for (int i = 0 ;i<size;i++){
        hostSrc0[i] = 0;
    }

    cudaSetDevice(0);
    cudaMalloc(&devSrc0, size * sizeof(int));
    cudaMemcpy((unsigned char *)devSrc0, (unsigned char *)hostSrc0,
                                         size * sizeof(int), cudaMemcpyHostToDevice);

    kernel<<<N_BLK, N_THRD_PER_BLK, 0>>>(devSrc0, 4);
    kernel<<<N_BLK, N_THRD_PER_BLK, 0>>>(devSrc0, 5);
    kernel<<<N_BLK, N_THRD_PER_BLK, 0>>>(devSrc0, 10);
    kernel<<<N_BLK, N_THRD_PER_BLK, 0>>>(devSrc0, 20);

    cudaDeviceSynchronize();   

    // cudaMalloc(blockHeaderDev, BLK_HDR_SIZE);
    // cudaMemcpy(*blockHeaderDev, (unsigned char*)block, BLK_HDR_SIZE, cudaMemcpyHostToDevice);

    // cudaMalloc(nonceValidDev, sizeof(int));
    // cudaMemset(*nonceValidDev, 0, sizeof(int));




    return 0;
}