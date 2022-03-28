#pragma once

#include "kmer_t.hpp"
#include <upcxx/upcxx.hpp>

#include <assert.h>
#include <vector>
#include <iostream>

#define NEW

struct FB{
    char f,b;
};

#ifdef NEW
struct HashMap{
    using dtable = upcxx::dist_object<std::vector<std::vector<kmer_pair>>>;

    dtable dist_table;

    int num_procs;
    int num_bits;

    uint64_t size;

    HashMap(size_t size);

    int get_target_rank(const pkmer_t &key_kmer);
    // Distributed insertion
    bool insert(const kmer_pair &kmer);
    // Distributed find
    bool find(const pkmer_t &key_kmer, kmer_pair &val_kmer);

    static void local_insert(dtable &table, const kmer_pair &kmer_, uint32_t bucket_idx);
    static FB local_find(dtable &table, const pkmer_t &key_kmer_, uint32_t bucket_idx);

    upcxx::future<std::vector<FB>> find_many(
            int target, const std::vector<pkmer_t> &key_kmers);

    static std::vector<FB> local_find_many(
        dtable &table, const std::vector<pkmer_t> &key_kmers, int num_bits, uint64_t size);

    upcxx::future<> insert_many(int target, const std::vector<kmer_pair> &kmers);
    static void local_insert_many(dtable &table, const std::vector<kmer_pair> &kmers, int num_bits, uint64_t size);
};

int log2(int x){
    assert(x != 0);
    int bits = 1;
    while((x >>= 1) > 0){
        bits++;
    };
    // Ex. x=1000 will only use the last 3 bits
    // Ex. x=1010 will use the last 4 bits
    if((1 << bits) == x) bits--;
    return bits;
};


HashMap::HashMap(size_t size_): dist_table({}) {
    this->size = size_;
    this->num_procs = upcxx::rank_n();
    // how many bits are used to get target rank
    this->num_bits = log2(num_procs);

    /* std::cout << "num_procs: " << this->num_procs << std::endl; */
    /* std::cout << "num_bits: " << this->num_bits << std::endl; */

    // resize to the given bucket size
    dist_table->resize(size_);
    // make sure all processes have finished building the hash table
    upcxx::barrier();
};

int HashMap::get_target_rank(const pkmer_t &key_kmer){
    return key_kmer.hash() % ((uint64_t)num_procs);
};

bool HashMap::insert(const kmer_pair& kmer){
    // these bits are used to identify target rank
    uint64_t h = kmer.hash();
    // NOTE: this assume no duplicate key will be inserted!
    upcxx::future<> fut = upcxx::rpc(h % (uint64_t)num_procs,
            HashMap::local_insert, dist_table, kmer, (h >> num_bits) % size);
    fut.wait();
    return true;
};

void HashMap::local_insert(dtable &table, const kmer_pair &kmer_, uint32_t bucket_idx){

    (*table)[bucket_idx].push_back(kmer_);

    return;
};

bool HashMap::find(const pkmer_t &key_kmer, kmer_pair& val_kmer){
    uint64_t h = key_kmer.hash();

    upcxx::future<FB> fut = upcxx::rpc(h % (uint64_t)num_procs,
            HashMap::local_find, dist_table, key_kmer, (h >> num_bits) % size);

    val_kmer.kmer = key_kmer;
    FB fb = fut.wait();

    val_kmer.fb_ext[0] = fb.b;
    val_kmer.fb_ext[1] = fb.f;
    return true;
};

FB HashMap::local_find(dtable &table, const pkmer_t &key_kmer_, uint32_t bucket_idx){
    std::vector<kmer_pair> &v = (*table)[bucket_idx];
    FB ret;

    // Resolve hash collision by searching
    for(kmer_pair &kp: v){
        if(kp.kmer == key_kmer_){
            ret.f = kp.fb_ext[1];
            ret.b = kp.fb_ext[0];
            break;
        }
    }
    return ret; // only need to return (f, b) pair
}

upcxx::future<std::vector<FB>> HashMap::find_many(int target, const std::vector<pkmer_t> &key_kmers){
    upcxx::future<std::vector<FB>> fut = upcxx::rpc(
        target, HashMap::local_find_many, dist_table, key_kmers, num_bits, size);
    return fut;
};

std::vector<FB> HashMap::local_find_many(
        dtable &table, const std::vector<pkmer_t> &key_kmers, int num_bits, uint64_t size){
    int n = key_kmers.size();
    std::vector<FB> ret(n);

    for(int i = 0; i < n; ++i){
        const pkmer_t &key_kmer = key_kmers[i];
        uint32_t bucket_idx = (key_kmer.hash() >> num_bits) % size;

        std::vector<kmer_pair> &v = (*table)[bucket_idx];
        // Resolve collision
        for(kmer_pair &kp: v){
            if(kp.kmer == key_kmer){
                ret[i].b = kp.fb_ext[0];
                ret[i].f = kp.fb_ext[1];
                break;
            }
        }
    }
    return ret;
};

upcxx::future<> HashMap::insert_many(int target, const std::vector<kmer_pair> &kmers){
    return upcxx::rpc(target,
            HashMap::local_insert_many, dist_table, kmers, num_bits, size);
};

void HashMap::local_insert_many(dtable &table, const std::vector<kmer_pair> &kmers, int num_bits, uint64_t size){
    // batch insertion
    for(const kmer_pair &kmer: kmers){
        uint32_t bucket_idx = (kmer.hash() >> num_bits) % size;
        (*table)[bucket_idx].push_back(kmer);
    }
    return;
};

#else
struct HashMap {
    std::vector<kmer_pair> data;
    std::vector<int> used;

    size_t my_size;

    size_t size() const noexcept;

    HashMap(size_t size);

    // Most important functions: insert and retrieve
    // k-mers from the hash table.
    bool insert(const kmer_pair& kmer);
    bool find(const pkmer_t& key_kmer, kmer_pair& val_kmer);

    // Helper functions

    // Write and read to a logical data slot in the table.
    void write_slot(uint64_t slot, const kmer_pair& kmer);
    kmer_pair read_slot(uint64_t slot);

    // Request a slot or check if it's already used.
    bool request_slot(uint64_t slot);
    bool slot_used(uint64_t slot);
};

HashMap::HashMap(size_t size) {
    my_size = size;
    data.resize(size);
    used.resize(size, 0);
}

bool HashMap::insert(const kmer_pair& kmer) {
    uint64_t hash = kmer.hash();
    uint64_t probe = 0;
    bool success = false;
    do {
        uint64_t slot = (hash + probe++) % size();
        success = request_slot(slot);
        if (success) {
            write_slot(slot, kmer);
        }
    } while (!success && probe < size());
    return success;
}

bool HashMap::find(const pkmer_t& key_kmer, kmer_pair& val_kmer) {
    uint64_t hash = key_kmer.hash();
    uint64_t probe = 0;
    bool success = false;
    do {
        uint64_t slot = (hash + probe++) % size();
        if (slot_used(slot)) {
            val_kmer = read_slot(slot);
            if (val_kmer.kmer == key_kmer) {
                success = true;
            }
        }
    } while (!success && probe < size());
    return success;
}

bool HashMap::slot_used(uint64_t slot) { return used[slot] != 0; }

void HashMap::write_slot(uint64_t slot, const kmer_pair& kmer) { data[slot] = kmer; }

kmer_pair HashMap::read_slot(uint64_t slot) { return data[slot]; }

bool HashMap::request_slot(uint64_t slot) {
    if (used[slot] != 0) {
        return false;
    } else {
        used[slot] = 1;
        return true;
    }
}

size_t HashMap::size() const noexcept { return my_size; }
#endif
