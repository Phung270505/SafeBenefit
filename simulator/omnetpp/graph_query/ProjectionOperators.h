#pragma once
#include <cstdint>
#include <cstring>
#include <istream>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

#pragma pack(push, 1)
struct EdgeRow { uint64_t src, dst; int64_t timestamp; uint32_t type; };
struct UpdateRow { uint64_t sequence; EdgeRow edge; };
#pragma pack(pop)
static_assert(sizeof(EdgeRow) == 28, "binary projection layout mismatch");
static_assert(sizeof(UpdateRow) == 36, "binary update layout mismatch");
struct Query {
    std::string id, kind;
    uint64_t seed;
    int64_t begin, end;
    uint64_t graphVersion;
    uint32_t groupModulus;
    uint32_t samplingDenominator;
    uint64_t graphVertexCount = 0;
    uint64_t graphEdgeCount = 0;
    uint64_t scanRows = 0;
    std::vector<uint64_t> scanWorkByPartition, scanRowsByPartition;
};

inline uint32_t transferSignature(const EdgeRow& edge) {
    uint64_t hash = 14695981039346656037ULL;
    auto mix = [&](uint64_t value, int bytes) {
        for (int i = 0; i < bytes; ++i) {
            hash ^= (value >> (8 * i)) & 255ULL;
            hash *= 1099511628211ULL;
        }
    };
    mix(edge.src, 8);
    mix(edge.dst, 8);
    mix(static_cast<uint64_t>(edge.timestamp), 8);
    mix(edge.type, 4);
    return static_cast<uint32_t>(hash ^ (hash >> 32));
}

inline uint32_t transferLabel(const EdgeRow& edge, uint32_t modulus) {
    if (modulus == 0) throw std::invalid_argument("group modulus must be positive");
    return transferSignature(edge) % modulus;
}

inline void selectProjection(const std::vector<EdgeRow>& edges, const Query& query,
                             std::vector<uint64_t>& selected) {
    selected.clear();
    for (const auto& edge : edges) {
        if (query.samplingDenominator == 0)
            throw std::invalid_argument("sampling denominator must be positive");
        if (transferSignature(edge) % query.samplingDenominator != 0) continue;
        if (query.kind == "CQ7" || query.kind == "NEIGHBOR_TYPE_AGG") {
            if (edge.src == query.seed) selected.push_back(transferLabel(edge, query.groupModulus));
            if (edge.dst == query.seed) selected.push_back(transferLabel(edge, query.groupModulus));
        } else if ((query.kind == "CQ9" || query.kind == "TEMPORAL_TYPE_AGG")
                   && query.begin <= edge.timestamp && edge.timestamp <= query.end)
            selected.push_back(transferLabel(edge, query.groupModulus));
    }
}

inline void aggregateProjection(const std::vector<uint64_t>& selected, std::map<uint64_t, uint64_t>& counts) {
    counts.clear();
    for (auto type : selected) counts[type]++;
}

inline uint64_t projectionChecksum(const std::vector<uint64_t>& selected) {
    uint64_t hash = 14695981039346656037ULL;
    for (auto value : selected) for (int i = 0; i < 8; ++i) {
        hash ^= (value >> (8 * i)) & 255; hash *= 1099511628211ULL;
    }
    return hash;
}

inline void appendVarint(std::string& output, uint64_t value) {
    while (value >= 0x80) {
        output.push_back(static_cast<char>((value & 0x7f) | 0x80));
        value >>= 7;
    }
    output.push_back(static_cast<char>(value));
}

inline uint64_t readVarint(const std::string& input, size_t& offset) {
    uint64_t value = 0;
    int shift = 0;
    for (int index = 0; index < 10; ++index) {
        if (offset >= input.size()) throw std::runtime_error("truncated compressed varint");
        const uint8_t byte = static_cast<uint8_t>(input[offset++]);
        value |= static_cast<uint64_t>(byte & 0x7f) << shift;
        if ((byte & 0x80) == 0) return value;
        shift += 7;
    }
    throw std::runtime_error("compressed varint exceeds uint64");
}

inline std::string encodeCompressed(
    const std::vector<std::pair<uint64_t, uint64_t>>& entries
) {
    constexpr size_t headerSize = 256;
    std::string output(headerSize, '\0');
    output.replace(0, 4, "SBC1", 4);
    const uint32_t count = static_cast<uint32_t>(entries.size());
    for (int index = 0; index < 4; ++index)
        output.push_back(static_cast<char>((count >> (8 * index)) & 0xff));
    uint64_t previous = 0;
    for (const auto& item : entries) {
        if (item.first < previous || item.second == 0)
            throw std::runtime_error("invalid compressed dictionary entry");
        appendVarint(output, item.first - previous);
        appendVarint(output, item.second);
        previous = item.first;
    }
    return output;
}

inline std::map<uint64_t, uint64_t> decodeCompressed(const std::string& input) {
    constexpr size_t headerSize = 256;
    if (input.size() < headerSize + 4 || input.compare(0, 4, "SBC1") != 0)
        throw std::runtime_error("invalid compressed payload header");
    for (size_t index = 4; index < headerSize; ++index)
        if (input[index] != '\0') throw std::runtime_error("invalid compressed reserved header");
    uint32_t count = 0;
    for (int index = 0; index < 4; ++index)
        count |= static_cast<uint32_t>(static_cast<uint8_t>(input[headerSize + index])) << (8 * index);
    size_t offset = headerSize + 4;
    uint64_t previous = 0;
    std::map<uint64_t, uint64_t> result;
    for (uint32_t index = 0; index < count; ++index) {
        const uint64_t label = previous + readVarint(input, offset);
        const uint64_t value = readVarint(input, offset);
        if (value == 0) throw std::runtime_error("compressed count is zero");
        result[label] += value;
        previous = label;
    }
    if (offset != input.size()) throw std::runtime_error("trailing compressed payload bytes");
    return result;
}

// Streaming equivalent of decodeCompressed, preserving validation and canonical labels.
template<class Consumer> inline void decodeCompressedStream(std::istream& input, Consumer consume) {
    char header[260];
    if (!input.read(header,260) || std::string(header,4) != "SBC1")
        throw std::runtime_error("invalid compressed header");
    for (size_t i=4;i<256;++i) if (header[i] != 0) throw std::runtime_error("invalid reserved header");
    uint32_t entries=0;
    for (int i=0;i<4;++i) entries |= static_cast<uint32_t>(static_cast<uint8_t>(header[256+i])) << (8*i);
    auto varint = [&]() {
        uint64_t value=0;
        for (int i=0;i<10;++i) {
            const int raw=input.get();
            if (raw==EOF) throw std::runtime_error("truncated varint");
            const uint8_t byte=static_cast<uint8_t>(raw);
            if (i==9 && byte>1) throw std::runtime_error("varint overflows uint64");
            value |= static_cast<uint64_t>(byte & 127) << (7*i);
            if (byte<128) return value;
        }
        throw std::runtime_error("varint exceeds uint64");
    };
    uint64_t previous=0;
    for (uint32_t i=0;i<entries;++i) {
        const uint64_t delta=varint(), count=varint();
        if (UINT64_MAX-previous<delta || count==0) throw std::runtime_error("invalid compressed entry");
        previous+=delta; consume(previous,count);
    }
    if (input.peek()!=EOF) throw std::runtime_error("trailing compressed bytes");
}
