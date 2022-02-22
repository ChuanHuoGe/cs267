#include "common.h"
#include <mpi.h>

#include <vector>
#include <bits/stdc++.h>

#define BINSIZE (cutoff * 2.1)

#define MIN(x,y) ((x)<(y)?(x):(y))

// Put any static global variables here that you will use throughout the simulation.
std::vector<std::vector<particle_t>> local_bins;

// rank_grids_send[rank] = the grids I need to send to you
// NOTE: I think the order is vital otherwise Isend, Irecv might be wrong
std::map<int, std::vector<int>> rank_grids_send;
std::map<int, std::vector<int>> rank_grids_recv;

std::map<int, std::vector<std::array<MPI_Request, 2>>> sendreqs;
std::map<int, std::vector<std::array<MPI_Request, 2>>> recvreqs;

// dim x dim
int dim;
int dim_square;
int q, r;
// local_bins
int local_offset;
int num_bins;
int num_bins_w_neighbors;

int dir[9][2] = {{-1, -1}, {-1, 0}, {-1, 1}, {0, -1}, {0, 0}, {0, 1}, {1, -1}, {1, 0}, {1, 1}};

int get_global_bin_idx(double x, double y){
    return floor(x / BINSIZE) * dim + floor(y / BINSIZE);
}

constexpr int get_bin_i(double x){
    return floor(x / BINSIZE);
}
constexpr int get_bin_j(double y){
    return floor(y / BINSIZE);
}

int get_rank_from_bin_idx(int bidx){
    // check if it belongs to the first r processes
    return (bidx < r * (q + 1))? (bidx / (q+1)):((bidx - r * (q + 1)) / q + r);
}

void apply_force_bidir(particle_t& particle, particle_t& neighbor) {
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

void init_simulation(particle_t* parts, int num_parts, double size, int rank, int num_procs) {
	// You can use this space to initialize data objects that you may need
	// This function will be called once before the algorithm begins
	// Do not do any particle simulation here

    dim = floor(size / BINSIZE) + 1;
    dim_square = dim * dim;

    q = dim_square / num_procs;
    r = dim_square % num_procs; // the first `r` process get 1 additional grid

    if(rank < r){
        local_offset = rank * (q+1);
    }else{
        local_offset = rank * q + r * (q+1);
    }

    num_bins = (rank < r)? (q+1):(q);

    // NOTE: sorted
    std::unordered_map<int, std::set<int>> temp_rank_grids_send;
    std::unordered_map<int, std::set<int>> temp_rank_grids_recv;

    for(int i = 0; i < num_bins; ++i){
        int bi = (local_offset + i) / dim;
        int bj = (local_offset + i) % dim;
        for(int d = 0; d < 4; ++d){
            int ni = bi + dir[d][0];
            int nj = bj + dir[d][1];
            if(ni < 0 or ni >= dim or nj < 0 or nj >= dim)
                continue;
            int nei_rank = get_rank_from_bin_idx(ni * dim + nj);
            // I need to send this grid to this process
            temp_rank_grids_send[nei_rank].insert(local_offset + i);
        }
        for(int d = 5; d < 9; ++d){
            int ni = bi + dir[d][0];
            int nj = bj + dir[d][1];
            if(ni < 0 or ni >= dim or nj < 0 or nj >= dim)
                continue;
            int nei_rank = get_rank_from_bin_idx(ni * dim + nj);
            // I want to receive (ni, nj) from `nei_rank`
            temp_rank_grids_recv[nei_rank].insert(ni * dim + nj);
        }
    }

    // Initialize reqs
    for(auto &kv: temp_rank_grids_send){
        int trank = kv.first;
        auto &grid = kv.second;
        for(int i: grid){
            rank_grids_send[trank].push_back(i);
        }

        sendreqs[trank].resize(grid.size());
    }
    for(auto &kv: temp_rank_grids_recv){
        int trank = kv.first;
        auto &grid = kv.second;
        for(int i: grid){
            rank_grids_recv[trank].push_back(i);
        }

        recvreqs[trank].resize(grid.size());
    }

    int last_bi = (local_offset + num_bins - 1) / dim;
    int last_bj = (local_offset + num_bins - 1) % dim;

    // the last (bi, bj)'s fartherest neighbor grid
    int farthest_bidx = MIN(dim-1, last_bi + 1) * dim + MIN(dim-1, last_bj + 1);
    num_bins_w_neighbors = farthest_bidx - local_offset;
    // Resize it to include the neighbors on the bottom or right
    local_bins.resize(num_bins_w_neighbors);

    // TODO: Put particles into bins
    // Can this be parallelized?
    for(int i = 0; i < num_parts; ++i){
        int bidx = get_global_bin_idx(parts[i].x, parts[i].y);
        if(local_offset <= bidx and bidx < local_offset + num_bins){
            local_bins[bidx - local_offset].push_back(parts[i]);
        }
    }
    // Make sure all the process collects the particles it need to compute forces
    MPI_Barrier(MPI_COMM_WORLD);
}


void send_grid(int target_rank,
        std::vector<particle_t> &grid,
        std::array<MPI_Request, 2> &reqs){

    int n = grid.size();
    MPI_Isend(&n, 1, MPI_INT, target_rank, 0, MPI_COMM_WORLD, &reqs[0]);
    MPI_Isend(&grid[0], n, PARTICLE, target_rank, 0, MPI_COMM_WORLD, &reqs[1]);
}

void recv_grid(int src_rank,
        std::vector<particle_t> &grid,
        std::array<MPI_Request, 2> &reqs){

    // Clear the previous step's particles
    grid.clear();

    int n;
    MPI_Recv(&n, 1, MPI_INT, src_rank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    MPI_Irecv(&grid[0], n, PARTICLE, src_rank, 0, MPI_COMM_WORLD, &reqs[1]);
}

void send_axay(int target_rank,
        std::vector<particle_t> &grid,
        std::array<MPI_Request, 2> &reqs){
    int n = grid.size();
    MPI_Isend(&grid[0], n, PARTICLE, target_rank, 0, MPI_COMM_WORLD, &reqs[1]);
}

void recv_axay(int src_rank,
        std::vector<particle_t> &grid,
        std::array<MPI_Request, 2> &reqs){

    std::vector<particle_t> tmp;
    // we already know the size, so no need to send n
    int n = grid.size();
    tmp.resize(n);
    MPI_Recv(&tmp[0], n, PARTICLE, src_rank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    for(int i = 0; i < n; ++i){
        grid[i].ax += tmp[i].ax;
        grid[i].ay += tmp[i].ay;
    }
}

void simulate_one_step(particle_t* parts, int num_parts, double size, int rank, int num_procs) {

    // 1. Each process collect and send the neighbor's (x, y) information

    // Send the information to top and left
    for(auto &kv: rank_grids_send){
        int target_rank = kv.first;
        auto &grid_indices = kv.second;
        int n = grid_indices.size();

        auto &reqs = sendreqs[target_rank];

        for(int i = 0; i < n; ++i){
            int bidx = grid_indices[i];
            send_grid(target_rank, local_bins[bidx - local_offset], reqs[i]);
        }
    }
    // Receive the information from bottom and right
    for(auto &kv: rank_grids_recv){
        int src_rank = kv.first;
        auto &grid_indices = kv.second;
        int n = grid_indices.size();

        auto &reqs = recvreqs[src_rank];

        for(int i = 0; i < n; ++i){
            int bidx = grid_indices[i];
            recv_grid(src_rank, local_bins[bidx - local_offset], reqs[i]);
        }
    }

    // Wait
    for(auto &kv: sendreqs){
        auto &reqs = kv.second;
        for(auto &req: reqs){
            MPI_Wait(&req[0], MPI_STATUS_IGNORE);
            MPI_Wait(&req[1], MPI_STATUS_IGNORE);
        }
    }
    for(auto &kv: recvreqs){
        auto &reqs = kv.second;
        for(auto &req: reqs){
            MPI_Wait(&req[1], MPI_STATUS_IGNORE);
        }
    }

    // Clear the accelerations
    for(int i = 0; i < num_bins_w_neighbors; ++i){
        auto &grid = local_bins[i];
        for(particle_t &p: grid){
            p.ax = p.ay = 0;
        }
    }

    // Compute bidirectional forces
    for(int i = 0; i < num_bins; ++i){
        auto &grid = local_bins[i];
        int bi = (local_offset + i) / dim;
        int bj = (local_offset + i) % dim;
        for(int d = 5; d < 9; ++d){
            int ni = bi + dir[d][0];
            int nj = bj + dir[d][1];
            if(ni < 0 or ni >= dim or nj < 0 or nj >= dim)
                continue;
            int nidx = ni * dim + nj;
            auto &nei_grid = local_bins[nidx - local_offset];

            for(particle_t &p1: grid){
                for(particle_t &p2: nei_grid){
                    apply_force_bidir(p1, p2);
                }
            }
        }
    }
    // Send the (ax, ay) information to its bottom or right neighbors
    for(auto &kv: rank_grids_recv){
        int target_rank = kv.first;
        auto &grid_indices = kv.second;
        int n = grid_indices.size();

        auto &reqs = recvreqs[target_rank];

        for(int i = 0; i < n; ++i){
            int bidx = grid_indices[i];
            send_axay(target_rank, local_bins[bidx], reqs[i]);
        }
    }
    // Recv the (ax, ay) information from top or left neighbors
    for(auto &kv: rank_grids_send){
        int src_rank = kv.first;
        auto &grid_indices = kv.second;
        int n = grid_indices.size();

        auto &reqs = sendreqs[src_rank];
        for(int i = 0; i < n; ++i){
            int bidx = grid_indices[i];
            recv_axay(src_rank, local_bins[bidx], reqs[i]);
        }
    }

    // Wait for sending
    for(auto &kv: sendreqs){
        auto &reqs = kv.second;
        for(auto &req: reqs){
            MPI_Wait(&req[1], MPI_STATUS_IGNORE);
        }
    }

    // Move
    // Particle redistribution
    /* MPI_Alltoallv(); */
}

void gather_for_save(particle_t* parts, int num_parts, double size, int rank, int num_procs) {
    // Write this function such that at the end of it, the master (rank == 0)
    // processor has an in-order view of all particles. That is, the array
    // parts is complete and sorted by particle id.
}
