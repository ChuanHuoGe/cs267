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

    static void local_insert(dtable &table, const kmer_pair &kmer_, const uint64_t &h_, const uint64_t &size_);
    static FB local_find(dtable &table, const pkmer_t &key_kmer_, const uint64_t &h_, const uint64_t &size_);
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
    uint64_t h = kmer.hash();
    // these bits are used to identify target rank
    h >>= num_bits;
    // NOTE: this assume no duplicate key will be inserted!
    upcxx::future<> fut = upcxx::rpc(get_target_rank(kmer.kmer),
            HashMap::local_insert, dist_table, kmer, h, size);
    fut.wait();
    return true;
};

void HashMap::local_insert(dtable &table, const kmer_pair &kmer_, const uint64_t &h_, const uint64_t &size_){

    (*table)[h_ % size_].push_back(kmer_);

    return;
};

bool HashMap::find(const pkmer_t &key_kmer, kmer_pair& val_kmer){
    uint64_t h = key_kmer.hash();
    h >>= num_bits;

    upcxx::future<FB> fut = upcxx::rpc(get_target_rank(key_kmer),
            HashMap::local_find, dist_table, key_kmer, h, size);

    val_kmer.kmer = key_kmer;
    FB fb = fut.wait();

    val_kmer.fb_ext[0] = fb.b;
    val_kmer.fb_ext[1] = fb.f;
    return true;
};

FB HashMap::local_find(dtable &table, const pkmer_t &key_kmer_,
    const uint64_t &h_, const uint64_t &size_){
    std::vector<kmer_pair> &v = (*table)[h_ % size_];
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
