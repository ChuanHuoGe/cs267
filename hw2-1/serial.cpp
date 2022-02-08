#include "common.h"
#include <cmath>

#include <vector>
#include <algorithm>

#define BINSIZE (0.01 + 0.01)

constexpr int bi(double x){
    return floor(x / BINSIZE);
}
constexpr int bj(double y){
    return floor(y / BINSIZE);
}
// bin[i][j] = the particles
std::vector<std::vector<particle_t*>> bins;
int griddim;
int dir[9][2] = {{1, 0}, {-1, 0}, {0, 1}, {0, -1}, {1, 1}, {1, -1}, {-1, 1}, {-1, -1}, {0, 0}};

// Apply the force from neighbor to particle
void apply_force(particle_t& particle, particle_t& neighbor) {
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

// Integrate the ODE
void move(particle_t& p, double size) {
    // Slightly simplified Velocity Verlet integration
    // Conserves energy better than explicit Euler method
    p.vx += p.ax * dt;
    p.vy += p.ay * dt;

    double x_ori = p.x;
    double y_ori = p.y;

    p.x += p.vx * dt;
    p.y += p.vy * dt;

    // Bounce from walls
    while (p.x < 0 || p.x > size) {
        p.x = p.x < 0 ? -p.x : 2 * size - p.x;
        p.vx = -p.vx;
    }

    while (p.y < 0 || p.y > size) {
        p.y = p.y < 0 ? -p.y : 2 * size - p.y;
        p.vy = -p.vy;
    }

    // no change, return
    if(p.x == x_ori and p.y == y_ori)
        return;

    // the coordinate changes, update this particle to a correct bin
    std::vector<particle_t*> &grid = bins[bi(x_ori) * griddim + bj(y_ori)];
    int grid_n = grid.size();
    // delete the particle from the original grid
    // NOTE: grid_n should be constant if the density for each grid is a constant
    for(int i = 0; i < grid_n; ++i){
        if(grid[i] == &p){
            grid.erase(grid.begin() + i);
            break;
        }
    }
    // insert the particle to the correct grid
    bins[bi(p.x) * griddim + bj(p.y)].push_back(&p);
    return;
}


void init_simulation(particle_t* parts, int num_parts, double size) {
	// You can use this space to initialize static, global data objects
    // that you may need. This function will be called once before the
    // algorithm begins. Do not do any particle simulation here

    griddim = floor(size / BINSIZE) + 1;

    // Put particles into the bins
    bins = std::vector<std::vector<particle_t*>>(griddim * griddim);
    for(int i = 0; i < num_parts; ++i){
        double x = parts[i].x;
        double y = parts[i].y;

        bins[bi(x) * griddim + bj(y)].push_back(&parts[i]);
    }
}

void simulate_one_step(particle_t* parts, int num_parts, double size) {
    // Loop over all the particles
    for(int i = 0; i < num_parts; ++i){
        int bii = bi(parts[i].x);
        int bjj = bj(parts[i].y);
        // Clear the acceleration
        parts[i].ax = parts[i].ay = 0;
        // Check 9 neighbor grids (including itself)
        for(int d = 0; d < 9; d++){
            int bi_nei = bii + dir[d][0];
            int bj_nei = bjj + dir[d][1];
            // out of bound
            if(bi_nei < 0 or bi_nei >= griddim or bj_nei < 0 or bj_nei >= griddim)
                continue;
            std::vector<particle_t*> &grid = bins[bi_nei * griddim + bj_nei];
            int grid_n = grid.size();
            // Loop over the particles inside this grid
            for(int j = 0; j < grid_n; ++j){
                particle_t *neighbor = grid[j];
                apply_force(parts[i], *neighbor);
            }
        }
    }

    // Move Particles and update each particle's bin
    for (int i = 0; i < num_parts; ++i) {
        move(parts[i], size);
    }
}
