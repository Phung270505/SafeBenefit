#pragma once
#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <functional>
#include <map>
#include <memory>
#include <queue>
#include <stdexcept>
#include <string>
#include <vector>

// Portable temporary records: two little-endian uint64 values, never struct padding.
inline void spillWrite64(std::ostream& out, uint64_t value) {
    for (int i = 0; i < 8; ++i) out.put(static_cast<char>((value >> (8*i)) & 255));
    if (!out) throw std::runtime_error("cannot write temporary grouping artifact");
}
inline bool spillRead64(std::istream& in, uint64_t& value) {
    int byte = in.get();
    if (byte == EOF) return false;
    value = static_cast<uint8_t>(byte);
    for (int i = 1; i < 8; ++i) {
        byte = in.get();
        if (byte == EOF) throw std::runtime_error("truncated temporary record");
        value |= static_cast<uint64_t>(static_cast<uint8_t>(byte)) << (8*i);
    }
    return true;
}

class SpillCounts {
    std::map<uint64_t, uint64_t> memory;
    std::vector<std::string> runs;
    std::string base, merged;
    size_t budget = 64 * 1024 * 1024, cardinality = 0;
    bool finalized = false;
    void flush() {
        if (memory.empty()) return;
        const std::string path = base + ".run-" + std::to_string(runs.size());
        std::ofstream out(path, std::ios::binary | std::ios::trunc);
        for (const auto& item : memory) { spillWrite64(out,item.first); spillWrite64(out,item.second); }
        writtenBytes += memory.size() * 16;
        runs.push_back(path); memory.clear();
    }
public:
    uint64_t writtenBytes = 0, peakBufferBytes = 0;
    mutable uint64_t readBytes = 0;
    ~SpillCounts() { clear(); }
    void clear() {
        for (const auto& path : runs) std::filesystem::remove(path);
        if (!merged.empty()) std::filesystem::remove(merged);
        memory.clear(); runs.clear(); merged.clear(); finalized = false; cardinality = 0;
        writtenBytes = readBytes = peakBufferBytes = 0;
    }
    void configure(const std::string& path, size_t bytes) {
        clear(); base = path; budget = std::max<size_t>(64, bytes);
        std::filesystem::create_directories(std::filesystem::path(base).parent_path());
    }
    void add(uint64_t key, uint64_t count = 1) {
        if (finalized) throw std::runtime_error("cannot modify finalized grouping");
        memory[key] += count;
        peakBufferBytes = std::max<uint64_t>(peakBufferBytes, memory.size() * 64);
        if (memory.size() * 64 >= budget) flush();
    }
    size_t mergeRuns(const std::vector<std::string>& paths, const std::string& destination) {
        struct Item { uint64_t key, count; size_t stream; };
        auto greater = [](const Item& a,const Item& b) { return a.key > b.key; };
        std::priority_queue<Item,std::vector<Item>,decltype(greater)> heap(greater);
        std::vector<std::unique_ptr<std::ifstream>> inputs;
        auto next = [&](size_t i) {
            uint64_t key,count;
            if (spillRead64(*inputs[i],key)) {
                if (!spillRead64(*inputs[i],count)) throw std::runtime_error("missing spill count");
                readBytes+=16; heap.push({key,count,i});
            }
        };
        for (const auto& path : paths) {
            inputs.push_back(std::make_unique<std::ifstream>(path,std::ios::binary));
            if (!*inputs.back()) throw std::runtime_error("cannot read spill run");
            next(inputs.size()-1);
        }
        std::ofstream out(destination,std::ios::binary | std::ios::trunc);
        size_t unique=0;
        while (!heap.empty()) {
            auto item=heap.top(); heap.pop(); next(item.stream);
            uint64_t count=item.count;
            while (!heap.empty() && heap.top().key==item.key) {
                auto same=heap.top(); heap.pop(); count+=same.count; next(same.stream);
            }
            spillWrite64(out,item.key); spillWrite64(out,count); writtenBytes+=16; ++unique;
        }
        inputs.clear(); out.close();
        for (const auto& path : paths) std::filesystem::remove(path);
        return unique;
    }
    void finalize() {
        if (finalized) return;
        if (runs.empty()) { cardinality=memory.size(); finalized=true; return; }
        flush();
        size_t round=0;
        while (runs.size()>64) {
            std::vector<std::string> next;
            for (size_t i=0;i<runs.size();i+=64) {
                const size_t end=std::min(runs.size(),i+64);
                const std::vector<std::string> paths(runs.begin()+i,runs.begin()+end);
                const std::string path=base+".pass-"+std::to_string(round)+"-"+std::to_string(i);
                mergeRuns(paths,path); next.push_back(path);
            }
            runs=std::move(next); ++round;
        }
        merged=base+".merged"; cardinality=mergeRuns(runs,merged); runs.clear(); finalized=true;
    }
    size_t size() const {
        if (!finalized) throw std::runtime_error("grouping cardinality requested before finalization");
        return cardinality;
    }
    template<class Consumer> void forEach(Consumer consume) const {
        if (!finalized) throw std::runtime_error("grouping iterated before finalization");
        if (merged.empty()) { for (const auto& item : memory) consume(item.first,item.second); return; }
        std::ifstream in(merged, std::ios::binary);
        uint64_t key, count;
        while (spillRead64(in,key)) {
            if (!spillRead64(in,count)) throw std::runtime_error("missing merged count");
            readBytes += 16; consume(key,count);
        }
    }
};

// A bounded-memory sequence of payloads, replayable without re-encoding on retry.
class MessageSpool {
    std::string path;
    mutable std::fstream stream;
    size_t entries = 0;
public:
    mutable uint64_t fileOpenCount = 0;
    ~MessageSpool() { clear(); }
    void clear() {
        if (stream.is_open()) stream.close();
        if (!path.empty()) std::filesystem::remove(path);
        path.clear(); entries = 0;
    }
    void configure(const std::string& value) {
        clear(); path = value;
        std::filesystem::create_directories(std::filesystem::path(path).parent_path());
        stream.open(path,std::ios::binary | std::ios::in | std::ios::out | std::ios::trunc);
        if (!stream) throw std::runtime_error("cannot open message spool");
        ++fileOpenCount;
    }
    uint64_t append(const std::string& payload) {
        stream.clear(); stream.seekp(0,std::ios::end);
        const uint64_t offset = static_cast<uint64_t>(stream.tellp());
        spillWrite64(stream,payload.size()); stream.write(payload.data(),payload.size());
        if (!stream) throw std::runtime_error("cannot spool reply");
        ++entries; return offset;
    }
    std::string at(uint64_t offset) const {
        stream.flush(); stream.clear(); stream.seekg(offset);
        uint64_t length;
        if (!spillRead64(stream,length) || length > 65507) throw std::runtime_error("invalid spool offset");
        std::string payload(length,'\0');
        if (!stream.read(payload.data(),length)) throw std::runtime_error("truncated spool entry");
        return payload;
    }
    size_t size() const { return entries; }
    template<class Consumer> void forEach(Consumer consume) const {
        if (path.empty()) return;
        stream.flush(); ++fileOpenCount;
        std::ifstream in(path,std::ios::binary);
        uint64_t length;
        while (spillRead64(in,length)) {
            if (length > 65507) throw std::runtime_error("oversized reply spool record");
            std::string payload(length,'\0');
            if (!in.read(payload.data(),length)) throw std::runtime_error("truncated reply spool");
            consume(payload);
        }
    }
};
