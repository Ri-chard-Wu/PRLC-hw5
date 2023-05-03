#include <cmath>           
#include <fstream>           
#include <iomanip>           
#include <limits>           
#include <stdexcept>           
#include <string>           
#include <vector>           

using namespace std;

namespace param {

const int n_steps = 200000;
const double dt = 60;
const double eps = 1e-3;
const double G = 6.674e-11;

double gravity_device_mass(double m0, double t) {
    return m0 + 0.5 * m0 * fabs(sin(t / 6000));
}

const double planet_radius = 1e7;
const double missile_speed = 1e6;
double get_missile_cost(double t) { return 1e5 + 1e3 * t; }

}  // namespace param




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




void write_output(const char* filename, double min_dist, int hit_time_step,
    int gravity_device_id, double missile_cost) {
    std::ofstream fout(filename);
    fout << std::scientific
         << std::setprecision(std::numeric_limits<double>::digits10 + 1) << min_dist
         << '\n'
         << hit_time_step << '\n'
         << gravity_device_id << ' ' << missile_cost << '\n';
}



void run_step(int step, Input *input) {

    // compute accelerations
    int n = input->n;
    std::vector<double> ax(n), ay(n), az(n);

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {

            if (j == i) continue;

            double mj = input->bodyArray[j].m;
            if (input->bodyArray[j].isDevice == 1) {
                mj = param::gravity_device_mass(mj, step * param::dt);
            }

            double dx = input->bodyArray[j].qx - input->bodyArray[i].qx;
            double dy = input->bodyArray[j].qy - input->bodyArray[i].qy;
            double dz = input->bodyArray[j].qz - input->bodyArray[i].qz;

            double dist3 =
                pow(dx * dx + dy * dy + dz * dz + param::eps * param::eps, 1.5);

            ax[i] += param::G * mj * dx / dist3;    
            ay[i] += param::G * mj * dy / dist3;    
            az[i] += param::G * mj * dz / dist3;    
        }
    }


    // update velocities
    for (int i = 0; i < n; i++) {
        input->bodyArray[i].vx += ax[i] * param::dt;
        input->bodyArray[i].vy += ay[i] * param::dt;
        input->bodyArray[i].vz += az[i] * param::dt;
    }

    // update positions
    for (int i = 0; i < n; i++) {
        input->bodyArray[i].qx += input->bodyArray[i].vx * param::dt;
        input->bodyArray[i].qy += input->bodyArray[i].vy * param::dt;
        input->bodyArray[i].qz += input->bodyArray[i].vz * param::dt;
    }
}



int main(int argc, char** argv) {

    if (argc != 3) {
        throw std::runtime_error("must supply 2 arguments");
    }

    int n, planet, asteroid;
    std::vector<double> qx, qy, qz, vx, vy, vz, m;
    std::vector<std::string> type;
    double dx, dy, dz, travel_dist;

    auto distance = [&](int i, int j) -> double {
        dx = qx[i] - qx[j];
        dy = qy[i] - qy[j];
        dz = qz[i] - qz[j];
        return sqrt(dx * dx + dy * dy + dz * dz);
    };


    // Problem 1
    double min_dist = std::numeric_limits<double>::infinity();

    // read_input(argv[1], n, planet, asteroid, qx, qy, qz, vx, vy, vz, m, type);

    Input input;
    read_input(argv[1], &input);

    for (int i = 0; i < input.n; i++) {
        if (input.bodyArray[i].isDevice == 1) input.bodyArray[i].m = 0;
    }

    for (int step = 0; step <= param::n_steps; step++) {
        if (step > 0) {
            run_step(step, &input);
        }
        dx = input.bodyArray[input.planetId].qx - input.bodyArray[input.asteroidId].qx;
        dy = input.bodyArray[input.planetId].qy - input.bodyArray[input.asteroidId].qy;
        dz = input.bodyArray[input.planetId].qz - input.bodyArray[input.asteroidId].qz;
        min_dist = std::min(min_dist, sqrt(dx * dx + dy * dy + dz * dz));
    }

    printf("min_dist: %f\n", min_dist);




    // // Problem 2
    // int hit_time_step = -2;
    // read_input(argv[1], n, planet, asteroid, qx, qy, qz, vx, vy, vz, m, type);
    // for (int step = 0; step <= param::n_steps; step++) {
    //     if (step > 0) {
    //         run_step(step, n, qx, qy, qz, vx, vy, vz, m, type);
    //     }
    //     dx = qx[planet] - qx[asteroid];
    //     dy = qy[planet] - qy[asteroid];
    //     dz = qz[planet] - qz[asteroid];
    //     if (dx * dx + dy * dy + dz * dz < param::planet_radius * param::planet_radius) {
    //         hit_time_step = step;
    //         break;
    //     }
    // }



    // // Problem 3
    // read_input(argv[1], n, planet, asteroid, qx, qy, qz, vx, vy, vz, m, type);

    // int gravity_device_id;
    // double missile_cost;  

    // int gravity_device_id_min = -1;
    // double missile_cost_min = std::numeric_limits<double>::infinity();

    // bool success;  

    // if(hit_time_step != -2){

    //     for(int i = 0; i < n; i++){

    //         if(type[i] != "device") continue;
    //         if(m[i] == 0) continue;

    //         gravity_device_id = i;
    //         success = true;

    //         for (int step = 0; step <= param::n_steps; step++) {
              
    //             if (step > 0) {
    //                 run_step(step, n, qx, qy, qz, vx, vy, vz, m, type);
    //             }
    //             // check missile hit device.
    //             dx = qx[planet] - qx[gravity_device_id];
    //             dy = qy[planet] - qy[gravity_device_id];
    //             dz = qz[planet] - qz[gravity_device_id];
    //             travel_dist = step * param::dt * param::missile_speed;
    //             if ((dx * dx + dy * dy + dz * dz < travel_dist * travel_dist)  \
    //                                                 && (m[gravity_device_id] != 0)) {
    //                 missile_cost = param::get_missile_cost((step + 1) * param::dt);
    //                 m[gravity_device_id] = 0;
                    
    //             }

    //             // check asteroid hit planet.
    //             dx = qx[planet] - qx[asteroid];
    //             dy = qy[planet] - qy[asteroid];
    //             dz = qz[planet] - qz[asteroid];
    //             if (dx * dx + dy * dy + dz * dz < param::planet_radius * param::planet_radius) {
    //                 success = false;
    //                 break;
    //             }
    //         }

    //         if(success){
    //             if(missile_cost < missile_cost_min){
    //                 missile_cost_min = missile_cost;
    //                 gravity_device_id_min = gravity_device_id;
    //             }
    //         }
    //         read_input(argv[1], n, planet, asteroid, qx, qy, qz, vx, vy, vz, m, type);
    //     }

    // }

    // if(gravity_device_id_min == -1){
    //     gravity_device_id = -1;
    //     missile_cost = 0;
    // }

    // write_output(argv[2], min_dist, hit_time_step, gravity_device_id, missile_cost);
}
