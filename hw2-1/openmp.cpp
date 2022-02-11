#include "common.h"
#include <omp.h>
#include <cmath>
#include <vector>

/* #include <iostream> */

#define BINSIZE (cutoff * 2.1)
#define NUMTHREADS 10

constexpr int bi(double x){
    return floor(x / BINSIZE);
}
constexpr int bj(double y){
    return floor(y / BINSIZE);
}

struct particle_pos{
    double x, y;
};

struct particle_t_w_addr{
    particle_t *addr; // for write back
    double x;  // Position X
    double y;  // Position Y
    double vx; // Velocity X
    double vy; // Velocity Y
    double ax; // Acceleration X
    double ay; // Acceleration Y
};

// Put any static global variables here that you will use throughout the simulation.
std::vector<std::vector<particle_t*>> bins;
int griddim;
/* int dir[9][2] = {{-1, -1}, {-1, 0}, {-1, 1}, {0, -1}, {0, 0}, {0, 1}, {1, -1}, {1, 0}, {1, 1}}; */
int dir[8][2] = {{-1, -1}, {-1, 0}, {-1, 1}, {0, -1}, {0, 1}, {1, -1}, {1, 0}, {1, 1}};

std::vector<particle_t_w_addr> cur[NUMTHREADS];

std::vector<omp_lock_t> bin_locks;

void apply_force(particle_t_w_addr& particle, particle_t& neighbor) {
    // Calculate Distance
    double dx = neighbor.x - particle.x;
    double dy = neighbor.y - particle.y;
    double r2 = dx * dx + dy * dy;

    // Check if the two particles should interact
    if (r2 > cutoff * cutoff)
        return;

    r2 = fmax(r2, min_r * min_r);
    double r = sqrt(r2);

    // Very simple short-range repulsive force
    double coef = (1 - cutoff / r) / r2 / mass;
    particle.ax += coef * dx;
    particle.ay += coef * dy;
}
void apply_force_bidir(particle_t_w_addr& particle, particle_t_w_addr& neighbor) {
    // Calculate Distance
    double dx = neighbor.x - particle.x;
    double dy = neighbor.y - particle.y;
    double r2 = dx * dx + dy * dy;

    // Check if the two particles should interact
    if (r2 > cutoff * cutoff)
        return;

    r2 = fmax(r2, min_r * min_r);
    double r = sqrt(r2);

    // Very simple short-range repulsive force
    double coef = (1 - cutoff / r) / r2 / mass;
    particle.ax += coef * dx;
    particle.ay += coef * dy;

    neighbor.ax -= coef * dx;
    neighbor.ay -= coef * dy;
}
// Integrate the ODE
void move(particle_t& p, double size) {
    // Slightly simplified Velocity Verlet integration
    // Conserves energy better than explicit Euler method
    double x_ori = p.x;
    double y_ori = p.y;

    // Update vx, vy and directly use there return value to update x,y
    p.x += (p.vx += p.ax * dt) * dt;
    p.y += (p.vy += p.ay * dt) * dt;

    // Bounce from walls
    while (p.x < 0 || p.x > size) {
        p.x = p.x < 0 ? -p.x : 2 * size - p.x;
        p.vx = -p.vx;
    }

    while (p.y < 0 || p.y > size) {
        p.y = p.y < 0 ? -p.y : 2 * size - p.y;
        p.vy = -p.vy;
    }

    // no change in bin
    if(bi(x_ori) == bi(p.x) and bj(y_ori) == bj(p.y)) return;

    const int from = bi(x_ori) * griddim + bj(y_ori);
    const int to = bi(p.x) * griddim + bj(p.y);

    // Lock the `from` grid
    omp_set_lock(&bin_locks[from]);
    // the coordinate changes, update this particle to a correct bin
    std::vector<particle_t*> &grid = bins[from];
    int grid_n = grid.size();
    // delete the particle from the original grid
    // NOTE: grid_n should be constant if the density for each grid is a constant
    for(int i = 0; i < grid_n; ++i){
        if(grid[i] == &p){
            grid.erase(grid.begin() + i);
            break;
        }
    }
    omp_unset_lock(&bin_locks[from]);

    // Lock the `to` grid
    omp_set_lock(&bin_locks[to]);
    // insert the particle to the correct grid
    bins[to].push_back(&p);
    omp_unset_lock(&bin_locks[to]);
    return;
}

void init_simulation(particle_t* parts, int num_parts, double size) {
	// You can use this space to initialize data objects that you may need
	// This function will be called once before the algorithm begins
	// Do not do any particle simulation here
    griddim = floor(size / BINSIZE) + 1;

    bins = std::vector<std::vector<particle_t*>>(griddim * griddim);
    bin_locks = std::vector<omp_lock_t>(griddim * griddim);
    const int space = ceil(1.2 * BINSIZE * BINSIZE * 1. / density);
    // Pre-reserve the memory at once
    for(int i = 0; i < griddim; ++i){
        for(int j = 0; j < griddim; ++j){
            // reserve the 1.2 * # of expected particles
            bins[i * griddim + j].reserve(space);
        }
    }
    // Put particles into the bins
    for(int i = 0; i < num_parts; ++i){
        double x = parts[i].x;
        double y = parts[i].y;

        bins[bi(x) * griddim + bj(y)].push_back(&parts[i]);
    }

    omp_set_num_threads(NUMTHREADS);
    /* std::cout << "num_threads:" << omp_get_max_threads() << std::endl; */
}

void simulate_one_step(particle_t* parts, int num_parts, double size) {

#pragma omp for collapse(2)
    for(int i = 0; i < griddim; ++i){
        for(int j = 0; j < griddim; ++j){
            int tid = omp_get_thread_num();
            // clean up
            cur[tid].clear();

            auto &center_grid = cur[tid];
            // Copy (i, j)
            auto &grid = bins[i * griddim + j];
            const int grid_n = grid.size();
            for(int k = 0; k < grid_n; ++k){
                particle_t *p = grid[k];
                // copy particle
                center_grid.push_back({.addr=p, .x=p->x, .y=p->y,
                        .vx=p->vx, .vy=p->vy, .ax=0, .ay=0});
            }
            // Apply force to itself
            for(int k = 0; k < grid_n; ++k){
                for(int l = k+1; l < grid_n; ++l){
                    apply_force_bidir(center_grid[k], center_grid[l]);
                }
            }
            // Apply neighbors' forces to the center grid
            for(int d = 0; d < 8; ++d){
                int ni = i + dir[d][0];
                int nj = j + dir[d][1];

                if(ni < 0 or ni >= griddim or nj < 0 or nj >= griddim)
                    continue;

                auto &nei = bins[ni * griddim + nj];
                const int nei_n = nei.size();

                for(int k = 0; k < nei_n; ++k){
                    particle_t *pp = nei[k];
                    // apply force to the center grid
                    for(int l = 0; l < grid_n; ++l){
                        apply_force(center_grid[l], *pp);
                    }
                }
            }
            // Write the acceleration back
            for(int k = 0; k < grid_n; ++k){
                particle_t_w_addr &p = center_grid[k];
                // NOTE: p.addr->ax will not be read by other threads
                // so it is fine to do these two operations
                p.addr->ax = p.ax;
                p.addr->ay = p.ay;
            }
        }
    }
    // implicit barrier

    // Move the particles and update the bins
#pragma omp for
    for(int i = 0; i < num_parts; ++i){
        move(parts[i], size);
    }
}
