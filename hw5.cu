
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




struct Body{
    double qx, qy, qz, vx, vy, vz, m;
    int isDevice;
};

struct Input{
    int n;
    int planetId;
    int asteroidId;
    Body *bodyArray;
};



__global__ void kernel_problem1(int n, int planetId, int asteroidId, Body *bodyArray){

    // compute accelerations
    std::vector<double> ax(n), ay(n), az(n);

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {

            if (j == i) continue;

            double mj = m[j];
            if (type[j] == "device") {
                mj = param::gravity_device_mass(mj, step * param::dt);
            }

            double dx = qx[j] - qx[i];
            double dy = qy[j] - qy[i];
            double dz = qz[j] - qz[i];

            double dist3 =
                pow(dx * dx + dy * dy + dz * dz + param::eps * param::eps, 1.5);

            ax[i] += param::G * mj * dx / dist3;    
            ay[i] += param::G * mj * dy / dist3;    
            az[i] += param::G * mj * dz / dist3;    
        }
    }


    // update velocities
    for (int i = 0; i < n; i++) {
        vx[i] += ax[i] * param::dt;
        vy[i] += ay[i] * param::dt;
        vz[i] += az[i] * param::dt;
    }

    // update positions
    for (int i = 0; i < n; i++) {
        qx[i] += vx[i] * param::dt;
        qy[i] += vy[i] * param::dt;
        qz[i] += vz[i] * param::dt;
    }
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
    double *bodyArray_dev;

    read_input(argv[1], &input);

    for (int i = 0; i < input.n; i++) {
        if (input.bodyArray[i].isDevice == 1) input.bodyArray[i].m = 0;
    }

    // printf("sizeof(Body): %d\n", sizeof(Body));

    cudaSetDevice(0);
    cudaMalloc(&bodyArray_dev, input.n * sizeof(Body));
    cudaMemcpy((unsigned char *)bodyArray_dev, (unsigned char *)input.bodyArray,
                                          input.n * sizeof(Body), cudaMemcpyHostToDevice);
    
     kernel_problem1<<<input.n, 32>>>(input.n, input.planetId, 
                                        input.asteroidId, bodyArray_dev);


















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