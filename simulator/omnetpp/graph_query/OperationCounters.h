#pragma once
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <ostream>
#include <stdexcept>

enum class Op : size_t { EdgeVisit, Predicate, Aggregate, Join, Tuple,
    EncodeByte, DecodeByte, Merge, Arithmetic, Comparison, Math, Control, Count };

struct OperationCounts {
    std::array<uint64_t, static_cast<size_t>(Op::Count)> counts{};
    void clear() { counts.fill(0); }
    void add(Op op, uint64_t n = 1) {
        auto& value = counts.at(static_cast<size_t>(op));
        if (std::numeric_limits<uint64_t>::max() - value < n)
            throw std::overflow_error("logical operation counter overflow");
        value += n;
    }
    uint64_t total() const {
        uint64_t sum = 0;
        for (auto value : counts) {
            if (std::numeric_limits<uint64_t>::max() - sum < value)
                throw std::overflow_error("logical work total overflow");
            sum += value;
        }
        return sum;
    }
    void accumulate(const OperationCounts& other) {
        for (size_t i = 0; i < counts.size(); ++i) add(static_cast<Op>(i), other.counts[i]);
    }
    void json(std::ostream& out) const {
        static const char* names[] = {"edge_visit", "predicate", "aggregate_update", "join_match",
            "tuple_visit", "encode_byte", "decode_byte", "merge_update", "arithmetic",
            "comparison", "special_math", "control_message"};
        out << '{';
        for (size_t i = 0; i < counts.size(); ++i) {
            if (i) out << ',';
            out << '\"' << names[i] << "\":" << counts[i];
        }
        out << '}';
    }
};

// Count each executed scalar operation; guards charge only evaluated branches.
struct CountedMath {
    OperationCounts& counter;
    double add(double a,double b) const { counter.add(Op::Arithmetic); return a+b; }
    double sub(double a,double b) const { counter.add(Op::Arithmetic); return a-b; }
    double mul(double a,double b) const { counter.add(Op::Arithmetic); return a*b; }
    double div(double a,double b) const { counter.add(Op::Arithmetic); return a/b; }
    double min(double a,double b) const { counter.add(Op::Comparison); return std::min(a,b); }
    double max(double a,double b) const { counter.add(Op::Comparison); return std::max(a,b); }
    bool less(double a,double b) const { counter.add(Op::Comparison); return a<b; }
    bool le(double a,double b) const { counter.add(Op::Comparison); return a<=b; }
    bool greater(double a,double b) const { counter.add(Op::Comparison); return a>b; }
    double log1p(double a) const { counter.add(Op::Math); return std::log1p(a); }
    double log2(double a) const { counter.add(Op::Math); return std::log2(a); }
    double log10(double a) const { counter.add(Op::Math); return std::log10(a); }
    double floor(double a) const { counter.add(Op::Math); return std::floor(a); }
    double ceil(double a) const { counter.add(Op::Math); return std::ceil(a); }
    double sqrt(double a) const { counter.add(Op::Math); return std::sqrt(a); }
    double hypot(double a,double b) const { counter.add(Op::Math); return std::hypot(a,b); }
};
