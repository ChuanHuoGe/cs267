#include "common.h"
#include <mpi.h>

#include <vector>
#include <bits/stdc++.h>
#include <iostream>
#include <assert.h>

#define BINSIZE (cutoff * 2.1)

#define MIN(x,y) (((x)<(y))?(x):(y))

// Put any static global variables here that you will use throughout the simulation.
std::vector<std::vector<particle_t>> local_bins;

// rank_grids_send[rank] = the grids I need to send to you
// NOTE: I think the order is vital otherwise Isend, Irecv might be wrong
std::map<int, std::vector<int>> rank_grids_send;
std::map<int, std::vector<int>> rank_grids_recv;

using Len = int;
std::map<int, std::vector<Len>> rank_grids_send_len;

std::map<int, std::vector<std::array<MPI_Request, 2>>> sendreqs;
std::map<int, std::vector<std::array<MPI_Request, 2>>> recvreqs;

// sendbuf[rank] == a list of buffers
std::map<int, std::vector<std::vector<char>>> sendbuf;

std::vector<int> num_send_parts;
std::vector<int> num_recv_parts;
std::vector<int> recv_displ;
std::vector<particle_t> local_parts_send;
std::vector<particle_t> local_parts_recv;

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

void move(particle_t& p, double size) {
    // Slightly simplified Velocity Verlet integration
    // Conserves energy better than explicit Euler method
    double x_ori = p.x;
    double y_ori = p.y;

    // Update vx, vy and directly use there return value to update x,y
    p.x += (p.vx += p.ax * dt) * dt;
    p.y += (p.vy += p.ay * dt) * dt;

    // reset (to denote that this particle has finished)
    p.ax = p.ay = 0;

    // Bounce from walls
    while (p.x < 0 || p.x > size) {
        p.x = p.x < 0 ? -p.x : 2 * size - p.x;
        p.vx = -p.vx;
    }

    while (p.y < 0 || p.y > size) {
        p.y = p.y < 0 ? -p.y : 2 * size - p.y;
        p.vy = -p.vy;
    }
    return;
}

void prt(int rank){
    std::cout << "Rank: " << rank << "\n";
    for(auto &kv: rank_grids_send){
        std::cout << "Send: \n";
        std::cout << "Key: " << kv.first << ", length: " << kv.second.size() << "\n";
    }
    for(auto &kv: rank_grids_recv){
        std::cout << "Recv: \n";
        std::cout << "Key: " << kv.first << ", length: " << kv.second.size() << "\n";
    }
    std::cout << "==============" << std::endl;
}


void init_simulation(particle_t* parts, int num_parts, double size, int rank, int num_procs) {
	// You can use this space to initialize data objects that you may need
	// This function will be called once before the algorithm begins
	// Do not do any particle simulation here

    dim = floor(size / BINSIZE) + 1;
    dim_square = dim * dim;

    // Each process will at least do `q` number of works
    q = dim_square / num_procs;
    /* if(rank == 0){ */
    /*     std::cout << "q: " << q << std::endl; */
    /*     std::cout << "dim_square: " << dim_square << std::endl; */
    /* } */
    // The first r processes will do 1 additional work
    r = dim_square % num_procs;

    if(rank < r){
        local_offset = rank * (q+1);
    }else{
        local_offset = r * (q+1) + (rank - r) * q;
    }

    // # of works(bins) to do
    num_bins = (rank < r)? (q+1):(q);

    // NOTE: sorted
    std::unordered_map<int, std::set<int>> temp_rank_grids_send;
    std::unordered_map<int, std::set<int>> temp_rank_grids_recv;

    // We need to for each process, which grid we need to send to him
    for(int i = 0; i < num_bins; ++i){
        int bi = (local_offset + i) / dim;
        int bj = (local_offset + i) % dim;
        for(int d = 0; d < 4; ++d){
            int ni = bi + dir[d][0];
            int nj = bj + dir[d][1];
            if(ni < 0 or ni >= dim or nj < 0 or nj >= dim)
                continue;
            int nei_rank = get_rank_from_bin_idx(ni * dim + nj);
            // If it is still in this process, skip
            if(nei_rank == rank)
                continue;
            // I need to send this grid to this process
            // NOTE: need to avoid duplicate grid indices
            temp_rank_grids_send[nei_rank].insert(local_offset + i);
        }
        for(int d = 5; d < 9; ++d){
            int ni = bi + dir[d][0];
            int nj = bj + dir[d][1];
            if(ni < 0 or ni >= dim or nj < 0 or nj >= dim)
                continue;
            int nei_rank = get_rank_from_bin_idx(ni * dim + nj);
            if(nei_rank == rank)
                continue;
            // I want to receive (ni, nj) from `nei_rank`
            temp_rank_grids_recv[nei_rank].insert(ni * dim + nj);
        }
    }

    // Initialize reqs
    for(auto &kv: temp_rank_grids_send){
        int target_rank = kv.first;
        auto &grid = kv.second;
        for(int i: grid){
            rank_grids_send[target_rank].push_back(i);
        }

        sendreqs[target_rank].resize(grid.size());
        rank_grids_send_len[target_rank].resize(grid.size());
    }
    for(auto &kv: temp_rank_grids_recv){
        int src_rank = kv.first;
        auto &grid = kv.second;
        for(int i: grid){
            rank_grids_recv[src_rank].push_back(i);
        }

        recvreqs[src_rank].resize(grid.size());
    }

    int last_bi = (local_offset + num_bins - 1) / dim;
    int last_bj = (local_offset + num_bins - 1) % dim;

    // the last (bi, bj)'s fartherest neighbor grid
    int farthest_bidx = MIN(dim-1, last_bi + 1) * dim + MIN(dim-1, last_bj + 1);
    // [local_offset, local_offset + num_bins - 1]: we are responsible for maintaining the particles
    // [local_offset + num_bins, local_offset + farthest_bidx]:
    // we are also responsible for computing forces for some of grids in this range
    num_bins_w_neighbors = farthest_bidx - local_offset + 1;
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


    num_send_parts.resize(num_procs);
    num_recv_parts.resize(num_procs);
    recv_displ.resize(num_procs);
    // at most `num_parts` (for root process)
    local_parts_send.reserve(num_parts);
    local_parts_recv.reserve(num_parts);
}


void send_grid(int target_rank,
        int &n,
        std::vector<particle_t> &grid,
        int bidx,
        std::array<MPI_Request, 2> &reqs
        ){

    // NOTE: n must be reference because n need to visible after this function ends
    MPI_Isend(&n, 1, MPI_INT, target_rank, 0, MPI_COMM_WORLD, &reqs[0]);
    MPI_Isend(grid.data(), n, PARTICLE, target_rank, 0, MPI_COMM_WORLD, &reqs[1]);
}

void recv_grid(int src_rank,
        std::vector<particle_t> &grid,
        int bidx,
        std::array<MPI_Request, 2> &reqs){

    // Clear the previous step's particles
    grid.clear();

    int n;
    MPI_Recv(&n, 1, MPI_INT, src_rank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    grid.resize(n);
    MPI_Irecv(grid.data(), n, PARTICLE, src_rank, 0, MPI_COMM_WORLD, &reqs[1]);
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

int sum(std::vector<int> &v){
    int s = 0;
    for(int e: v){
        s += e;
    }
    return s;
}

void exclusive_psum(std::vector<int> &v, std::vector<int> &p){
    int s = 0;
    int n = v.size();
    for(int i = 0; i < n; ++i){
        p[i] = s;
        s += v[i];
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

        auto &lens = rank_grids_send_len[target_rank];

        for(int i = 0; i < n; ++i){
            int bidx = grid_indices[i];
            lens[i] = local_bins[bidx - local_offset].size();
            // Use a reference because Isend
            int &num_parts_grid = lens[i];
            send_grid(target_rank, num_parts_grid, local_bins[bidx - local_offset], bidx, reqs[i]);
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
            recv_grid(src_rank, local_bins[bidx - local_offset], bidx, reqs[i]);
        }
    }

    // Wait to make sure you actually send it out
    for(auto &kv: sendreqs){
        auto &target_rank = kv.first;
        auto &reqs = kv.second;
        int n = reqs.size();

        for(int i = 0; i < n; ++i){
            auto &req = reqs[i];

            MPI_Wait(&req[0], MPI_STATUS_IGNORE);
            MPI_Wait(&req[1], MPI_STATUS_IGNORE);
        }
    }
    // Wait to make sure you actually receive it
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
        // Compute the forces within the grid
        int grid_n = grid.size();
        for(int j = 0; j < grid_n; ++j){
            for(int k = j+1; k < grid_n; ++k){
                apply_force_bidir(grid[j], grid[k]);
            }
        }
        // compute forces within this grid
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
            send_axay(target_rank, local_bins[bidx - local_offset], reqs[i]);
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
            recv_axay(src_rank, local_bins[bidx - local_offset], reqs[i]);
        }
    }

    // Wait for sending
    for(auto &kv: recvreqs){
        auto &reqs = kv.second;
        for(auto &req: reqs){
            MPI_Wait(&req[1], MPI_STATUS_IGNORE);
        }
    }

    // Move and collect those particles that need to be sent to other processes
    // need_dist[target_rank] = a list of particles I need to send to target_rank
    std::map<int, std::vector<particle_t>> need_redist;
    std::vector<particle_t> parts_moved_but_still_mine;
    for(int i = 0; i < num_bins; ++i){
        auto &grid = local_bins[i];
        auto it = grid.begin();
        int grid_n = grid.size();
        // Loop from the back because we are going to delete the particle
        // as we loop over it
        for(int j = grid_n-1; j >= 0; --j){
            // if this particle has been processed (because of bin redistributing)
            // continue
            particle_t &p = grid[j];
            // update (x, y, vx, vy) from (ax, ay) information
            // and reset (ax, ay)
            move(p, size);
            //
            int new_bidx = get_global_bin_idx(p.x, p.y);
            if(new_bidx == local_offset + i){
                // no move
                continue;
            }else if(local_offset <= new_bidx and new_bidx < local_offset + num_bins){
                // move, but still in this process (i.e. rank)
                parts_moved_but_still_mine.push_back(p);
                // delete this particle (that's why loop from the back)
                grid.erase(it + j);
            }else{
                // need to send this particle to the other process (i.e. rank)
                int target_rank = get_rank_from_bin_idx(new_bidx);
                need_redist[target_rank].push_back(p);
                // delete this particle
                grid.erase(it + j);
            }
        }
    }
    // Put into the right bin
    for(particle_t &p: parts_moved_but_still_mine){
        int bidx = get_global_bin_idx(p.x, p.y);
        local_bins[bidx - local_offset].push_back(p);
    }

    // --------------------------------------------------------
    // Particle redistribution
    int send_total = 0;
    std::vector<int> num_send_parts(num_procs, 0);
    for(auto &kv: need_redist){
        int target_rank = kv.first;
        auto &ps = kv.second;
        num_send_parts[target_rank] += ps.size();
        send_total += ps.size();
    }
    // num_recv_parts[i] == # of particles process i is gonna send me
    std::vector<int> num_recv_parts(num_procs);
    // Send: how many particles am I gonna send to each processor
    // Recv: how many particles will I expect to receive from each processor
    MPI_Alltoall(&num_send_parts[0], 1, MPI_INT,
                 &num_recv_parts[0], 1, MPI_INT,
                 MPI_COMM_WORLD);

    std::vector<particle_t> send_parts;
    send_parts.reserve(send_total);

    std::vector<int> send_displ(num_procs);

    int recv_total = sum(num_recv_parts);

    std::vector<particle_t> recv_parts(recv_total);
    std::vector<int> recv_displ(num_procs);

    // flatten
    for(auto &kv: need_redist){
        for(particle_t &t: kv.second){
            send_parts.push_back(t);
        }
    }
    // compute the displacement
    exclusive_psum(num_send_parts, send_displ);
    exclusive_psum(num_recv_parts, recv_displ);

    // Send and recv the particles
    // implicit barrier
    MPI_Alltoallv(&send_parts[0], &num_send_parts[0], &send_displ[0], PARTICLE,
                &recv_parts[0], &num_recv_parts[0], &recv_displ[0], PARTICLE,
                MPI_COMM_WORLD);

    // Put the recv_parts into the right bins
    for(particle_t &p: recv_parts){
        int bidx = get_global_bin_idx(p.x, p.y);
        local_bins[bidx - local_offset].push_back(p);
    }
    return;
}

void gather_for_save(particle_t* parts, int num_parts, double size, int rank, int num_procs) {
    // Write this function such that at the end of it, the master (rank == 0)
    // processor has an in-order view of all particles. That is, the array
    // parts is complete and sorted by particle id.

    // Each process tell the root process:
    // how many particles will I send you
    // and then gather the particles the root needs
    int num_parts_to_send = 0;

    local_parts_send.clear();

    // Flatten
    for(int i = 0; i < num_bins; ++i){
        int n = local_bins[i].size();
        num_parts_to_send += n;
        for(int j = 0; j < n; ++j){
            local_parts_send.push_back(local_bins[i][j]);
        }
    }

    if(rank == 0){
        MPI_Gather(&num_parts_to_send, 1, MPI_INT, num_recv_parts.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);
    }else{
        MPI_Gather(&num_parts_to_send, 1, MPI_INT, nullptr, 0, MPI_INT, 0, MPI_COMM_WORLD);
    }

    // count how many elements will I receive
    if(rank == 0){
        // the root expects to receive `total` number of particles
        int total = sum(num_recv_parts);
        local_parts_recv.resize(total);

        // Compute the displacement
        exclusive_psum(num_recv_parts, recv_displ);

        MPI_Gatherv(local_parts_send.data(), local_parts_send.size(),
            PARTICLE, local_parts_recv.data(),
            num_recv_parts.data(), recv_displ.data(),
            PARTICLE, 0, MPI_COMM_WORLD);

        // put pack all particles
        for(particle_t &p: local_parts_recv){
            parts[p.id-1] = p;
        }
    }else{
        MPI_Gatherv(local_parts_send.data(), local_parts_send.size(),
            PARTICLE, nullptr,
            nullptr, nullptr,
            PARTICLE, 0, MPI_COMM_WORLD);
    }
}
