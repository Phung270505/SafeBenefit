#include "OperationCounters.h"
#include "ProjectionOperators.h"
#include "SpillStore.h"
#include <cassert>
#include <sstream>
#include <iostream>

int main(int argc,char** argv) {
    if (argc != 2) return 2;
    const std::string root=argv[1];
    OperationCounts counts; counts.add(Op::Tuple,3); counts.add(Op::Arithmetic,2);
    assert(counts.total()==5);
    counts.clear(); CountedMath math{counts};
    assert(math.add(math.mul(2,3),math.div(8,4))==8);
    assert(counts.counts[static_cast<size_t>(Op::Arithmetic)]==3);
    assert(math.max(1,2)==2 && counts.counts[static_cast<size_t>(Op::Comparison)]==1);
    std::vector<std::pair<uint64_t,uint64_t>> expected={{0,2},{127,1},{128,3},{UINT64_MAX,1}};
    const std::string encoded=encodeCompressed(expected);
    std::istringstream input(encoded);
    std::vector<std::pair<uint64_t,uint64_t>> decoded;
    decodeCompressedStream(input,[&](uint64_t key,uint64_t count){decoded.push_back({key,count});});
    assert(decoded==expected);
    bool rejected=false;
    try { std::istringstream truncated(encoded.substr(0,encoded.size()-1)); decodeCompressedStream(truncated,[](auto,auto){}); }
    catch(const std::runtime_error&) {rejected=true;}
    assert(rejected);
    SpillCounts spill; spill.configure(root+"/counter",128);
    for(uint64_t i=0;i<1000;++i) {spill.add(i);spill.add(i);}
    spill.finalize(); assert(spill.size()==1000); assert(spill.peakBufferBytes<=128);
    uint64_t seen=0; spill.forEach([&](uint64_t key,uint64_t value){assert(key==seen++); assert(value==2);});
    assert(seen==1000 && spill.readBytes>0 && spill.writtenBytes>0);
    MessageSpool spool; spool.configure(root+"/messages");
    const auto offset=spool.append(std::string("a\0b",3));
    assert(spool.at(offset)==std::string("a\0b",3));
    for(int i=0;i<1000;++i) {
        const auto position=spool.append(std::string("a\0b",3));
        assert(spool.at(position)==std::string("a\0b",3));
    }
    assert(spool.fileOpenCount==1);
    spool.forEach([](const std::string& bytes){assert(bytes==std::string("a\0b",3));});
    std::cout << "REFERENCE_CORE_PASS\n";
}
