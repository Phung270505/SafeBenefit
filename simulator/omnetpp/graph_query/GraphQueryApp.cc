// SafeBenefit transfer suffix over INET UDP. Query operators may supply a
// materialized safe-point stream; candidate transfer payloads are still built
// only after the coordinator commits one physical plan.
#include <omnetpp.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <deque>
#include <fstream>
#include <iomanip>
#include <iterator>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <vector>
#include "ProjectionOperators.h"
#include "OperationCounters.h"
#include "SpillStore.h"
#include "inet/common/InitStages.h"
#include "inet/common/packet/Packet.h"
#include "inet/common/packet/chunk/BytesChunk.h"
#include "inet/networklayer/common/L3AddressResolver.h"
#include "inet/transportlayer/contract/udp/UdpSocket.h"

using namespace omnetpp;
using namespace inet;


class GraphQueryApp : public cSimpleModule, public UdpSocket::ICallback, public cListener {
    UdpSocket socket;
    bool coordinator = false;
    int worker = -1, workers = 4, queryIndex = -1, version = 0, plan = 0;
    int proposed = 0, phase = 0;
    std::string method;
    std::vector<EdgeRow> edges;
    std::vector<UpdateRow> updates;
    size_t updateCursor = 0;
    uint64_t appliedVersion = 0;
    std::vector<Query> queries;
    std::map<std::string, int> initialPlans;
    std::vector<uint64_t> selected;
    SpillCounts grouped, result;
    uint64_t committedChecksum = 0;
    std::map<int, uint64_t> rowCounts, distinctCounts, labelDigitSums, scannedEdgeCounts;
    std::map<int, double> serviceTimes, scanCpuTimes, scanIoTimes, scanQueueTimes, workerCpuRates;
    std::set<int> acknowledgements, rejected, completeWorkers;
    std::map<std::pair<int, int>, uint64_t> chunks;
    std::map<int, int> expectedChunks, receivedChunks;
    MessageSpool cachedReply, outboundSpool, receivedSpool;
    OperationCounts scanOps, encodeOps, decodeOps, mergeOps, protocolOps;
    mutable OperationCounts decisionOps;
    CountedMath decisionMath{decisionOps};
    uint64_t selectedCount = 0, committedLabelDigits = 0, localScannedRows = 0;
    double decisionCpuService = 0, encodeCpuService = 0, decodeCpuService = 0, suffixIoService = 0;
    double coordinatorQueueWait = 0;
    simtime_t coordinatorCpuAvailable;
    cMessage *decisionEvent = nullptr;
    std::string cachedReady;
    bool injectedDrop = false;
    bool cancelled = false;
    int retries = 0, decisionRetries = 0;
    struct OutboundMessage {
        int destination;
        std::string payload;
        bool retry;
        uint64_t spoolOffset = 0;
        bool spooled = false;
    };
    std::deque<OutboundMessage> outbound;
    cMessage *startEvent = nullptr, *serviceEvent = nullptr, *sendEvent = nullptr, *deadlineEvent = nullptr, *retryEvent = nullptr;
    cMessage *cancelQueryEvent = nullptr;
    simtime_t started, scanStarted, barrierAt;
    uint64_t sentBytes = 0, receivedBytes = 0, startSent = 0, startReceived = 0;
    uint64_t wireBytes = 0, startWire = 0, startSwitchWire = 0, switchWireBytes = 0;
    uint64_t retryWireBytes = 0, startRetryWireBytes = 0;
    uint64_t resultPayloadBytes = 0;
    std::vector<cComponent *> wireSources;
    double scanService = 0, scanCpuService = 0, scanIoService = 0, scanQueueService = 0;
    double spillBytes = 0, predictedCurrent = 0, predictedBest = 0, observedMeanLabel = 0;
    double observedPartitionSkewCv = 0;
    std::vector<double> predictedPlanBytes = {0, 0, 0};
    bool policyLoaded = false, policyFallback = false, policyForceFallback = false;
    std::vector<double> policyMean, policyScale, policyOodInverse;
    std::vector<std::vector<double>> policyTheta, policyPrecision;
    int policyInputSize = 0;
    double oodThreshold = 0, policyAlpha = 0;
    bool switchThresholdLoaded = false;
    double switchThreshold = 0;
    double estimatedSwitchWire = 0;
    uint64_t decisionLatencyNs = 0;
    std::string decisionReason;
    std::ofstream output;

    std::string spoolPath(const std::string& suffix) const {
        const std::string root = par("spoolDirectory").stdstringValue();
        if (root.empty()) throw cRuntimeError("spoolDirectory is required");
        return root + "/q" + std::to_string(queryIndex) + "-w" + std::to_string(worker) + "-" + suffix;
    }
    template<class Consumer> void visitSelected(Consumer consume) const {
        const auto& q = queries.at(queryIndex);
        if (!isMaterializedQuery(q)) { for (auto token : selected) consume(token); return; }
        const std::string root = par("materializedDirectory").stdstringValue();
        for (int part = 0; part < par("partitionCount").intValue(); ++part) if (part % workers == worker) {
            const std::string path = root + "/" + q.id + "/part-" + std::to_string(part) + ".bin";
            std::ifstream input(path, std::ios::binary);
            if (!input) throw cRuntimeError("missing materialized query partition");
            uint64_t token;
            while (spillRead64(input, token)) consume(token);
        }
    }
    uint64_t partitionSum(const std::vector<uint64_t>& values) const {
        uint64_t total = 0;
        for (size_t part = 0; part < values.size(); ++part)
            if ((int)part % workers == worker) total += values[part];
        return total;
    }
    GraphQueryApp* workerApp(int index) const {
        return check_and_cast<GraphQueryApp*>(getParentModule()->getParentModule()
            ->getSubmodule("worker", index)->getSubmodule("app", 0));
    }

    static std::vector<std::string> tabFields(const std::string& line) {
        std::vector<std::string> fields;
        std::stringstream stream(line);
        std::string field;
        while (std::getline(stream, field, '\t')) fields.push_back(field);
        return fields;
    }
    static std::vector<double> numericFields(const std::vector<std::string>& fields) {
        std::vector<double> values;
        for (size_t i = 1; i < fields.size(); ++i) values.push_back(std::stod(fields[i]));
        return values;
    }
    static bool finiteValues(const std::vector<double>& values) {
        return std::all_of(values.begin(), values.end(), [](double value) {
            return std::isfinite(value);
        });
    }
    std::vector<double> policyScores(const std::vector<double>& state) const {
        if (!policyLoaded || (int)state.size() != policyInputSize)
            throw cRuntimeError("invalid frozen policy inference request");
        std::vector<double> scores(policyTheta.size(), 0.0);
        std::vector<double> context(policyInputSize + 1, 1.0);
        std::copy(state.begin(), state.end(), context.begin() + 1);
        for (size_t action = 0; action < policyTheta.size(); ++action) {
            for (size_t column = 0; column < context.size(); ++column) {
                decisionOps.add(Op::Arithmetic, 2);
                scores[action] += policyTheta[action][column] * context[column];
            }
            double quadratic = 0;
            if (policyAlpha != 0) {
            for (size_t row = 0; row < context.size(); ++row)
                for (size_t column = 0; column < context.size(); ++column) {
                    decisionOps.add(Op::Arithmetic, 3);
                    quadratic += context[row]
                        * policyPrecision[action][row * context.size() + column]
                        * context[column];
                }
            decisionOps.add(Op::Math); decisionOps.add(Op::Arithmetic, 2); decisionOps.add(Op::Comparison);
            scores[action] += policyAlpha * std::sqrt(std::max(0.0, quadratic));
            }
        }
        return scores;
    }
    void loadPolicy(const std::string& path) {
        if (path.empty()) return;
        std::ifstream input(path);
        if (!input) throw cRuntimeError("cannot open frozen RL policy");
        std::string line;
        bool schema = false, methodName = false, exploration = false, fallbackGate = false;
        int actionCount = 0;
        while (std::getline(input, line)) {
            auto fields = tabFields(line);
            if (fields.empty()) continue;
            if (fields[0] == "schema_version") schema = fields.size() == 2 && fields[1] == "5";
            else if (fields[0] == "method") methodName = fields.size() == 2 && fields[1] == "LINUCB";
            else if (fields[0] == "mean") policyMean = numericFields(fields);
            else if (fields[0] == "scale") policyScale = numericFields(fields);
            else if (fields[0] == "ood_inverse_covariance") policyOodInverse = numericFields(fields);
            else if (fields[0] == "input_size" && fields.size() == 2) policyInputSize = std::stoi(fields[1]);
            else if (fields[0] == "action_count" && fields.size() == 2) {
                actionCount = std::stoi(fields[1]);
                policyTheta.resize(actionCount);
                policyPrecision.resize(actionCount);
            }
            else if (fields[0].rfind("theta_", 0) == 0) {
                const int action = std::stoi(fields[0].substr(6));
                if (action < 0 || action >= actionCount)
                    throw cRuntimeError("invalid frozen policy action row");
                policyTheta[action] = numericFields(fields);
            }
            else if (fields[0].rfind("precision_", 0) == 0) {
                const int action = std::stoi(fields[0].substr(10));
                if (action < 0 || action >= actionCount)
                    throw cRuntimeError("invalid frozen policy precision row");
                policyPrecision[action] = numericFields(fields);
            }
            else if (fields[0] == "ucb_alpha" && fields.size() == 2) policyAlpha = std::stod(fields[1]);
            else if (fields[0] == "ood_threshold" && fields.size() == 2) oodThreshold = std::stod(fields[1]);
            else if (fields[0] == "exploration_evaluation") exploration = fields.size() == 2 && fields[1] == "0";
            else if (fields[0] == "force_fallback" && fields.size() == 2) {
                if (fields[1] != "0" && fields[1] != "1")
                    throw cRuntimeError("invalid frozen policy fallback flag");
                policyForceFallback = fields[1] == "1";
                fallbackGate = true;
            }
        }
        policyLoaded = schema && methodName && exploration && fallbackGate && actionCount == 3
            && policyInputSize == 17 && policyMean.size() == 17 && policyScale.size() == 17
            && policyOodInverse.size() == 17 * 17
            && policyTheta.size() == 3 && policyPrecision.size() == 3
            && std::isfinite(policyAlpha) && policyAlpha >= 0 && oodThreshold > 0;
        for (double value : policyScale) if (!std::isfinite(value) || value <= 0) policyLoaded = false;
        if (!finiteValues(policyMean)) policyLoaded = false;
        if (!finiteValues(policyOodInverse)) policyLoaded = false;
        for (const auto& values : policyTheta)
            if (values.size() != 18 || !finiteValues(values)) policyLoaded = false;
        for (const auto& values : policyPrecision)
            if (values.size() != 18 * 18 || !finiteValues(values)) policyLoaded = false;
        if (!policyLoaded) throw cRuntimeError("invalid frozen RL policy contract");
    }
    void loadSwitchThreshold(const std::string& path) {
        if (path.empty()) return;
        std::ifstream input(path);
        if (!input) throw cRuntimeError("cannot open frozen switch-threshold model");
        std::string line;
        bool schema = false, methodName = false, frozen = false, thresholdFound = false;
        while (std::getline(input, line)) {
            auto fields = tabFields(line);
            if (fields.empty()) continue;
            if (fields[0] == "schema_version") schema = fields.size() == 2 && fields[1] == "1";
            else if (fields[0] == "method") methodName = fields.size() == 2 && fields[1] == "SWITCH_COST_THRESHOLD";
            else if (fields[0] == "threshold" && fields.size() == 2) {
                switchThreshold = std::stod(fields[1]);
                thresholdFound = true;
            }
            else if (fields[0] == "updates_evaluation") frozen = fields.size() == 2 && fields[1] == "0";
        }
        switchThresholdLoaded = schema && methodName && frozen && thresholdFound
            && std::isfinite(switchThreshold) && switchThreshold >= 0 && switchThreshold <= 1;
        if (!switchThresholdLoaded) throw cRuntimeError("invalid frozen switch-threshold contract");
    }
    std::vector<double> modelContext(
        uint64_t rows, uint64_t distinct, double meanLabel, double compressionRatio) const {
        const double logRows = decisionMath.log1p((double)rows);
        const double logDistinct = decisionMath.log1p((double)distinct);
        const double distinctRatio = rows ? decisionMath.div(distinct,rows) : 0.0;
        int queryClassId = 0;
        const std::string kind = queries[queryIndex].kind;
        if (kind == "TEMPORAL_TYPE_AGG") queryClassId = 1;
        else if (kind == "TWO_HOP_REACHABILITY") queryClassId = 2;
        else if (kind == "TWO_HOP_JOIN") queryClassId = 3;
        uint64_t graphScaleEdges = queries[queryIndex].graphEdgeCount;
        if (graphScaleEdges == 0)
            for (const auto& item : scannedEdgeCounts) graphScaleEdges += item.second;
        return {logRows, logDistinct, distinctRatio, meanLabel,
            compressionRatio, (double)plan, logRows - logDistinct,
            distinctRatio * meanLabel,
            queryClassId == 0 ? 1.0 : 0.0, queryClassId == 1 ? 1.0 : 0.0,
            queryClassId == 2 ? 1.0 : 0.0, queryClassId == 3 ? 1.0 : 0.0,
            decisionMath.log1p((double)graphScaleEdges), decisionMath.log1p((double)queries[queryIndex].graphVersion),
            decisionMath.log1p((double)workers), decisionMath.log1p((double)decisionRetries),
            decisionMath.log1p(estimatedSwitchWire)};
    }
    void applyUpdates(uint64_t targetVersion) {
        if (targetVersion < appliedVersion) throw cRuntimeError("graph version moved backwards");
        if (!par("materializedDirectory").stdstringValue().empty()) {
            appliedVersion = targetVersion;
            return;
        }
        while (updateCursor < updates.size() && updates[updateCursor].sequence < targetVersion) {
            edges.push_back(updates[updateCursor].edge);
            ++updateCursor;
        }
        appliedVersion = targetVersion;
    }

    uint64_t checksum() const {
        if (!isMaterializedQuery(queries.at(queryIndex))) return projectionChecksum(selected);
        uint64_t hash = 14695981039346656037ULL;
        visitSelected([&](uint64_t token) { for (int i = 0; i < 8; ++i) {
            hash ^= (token >> (8*i)) & 255; hash *= 1099511628211ULL;
        }});
        return hash;
    }
    bool isMaterializedQuery(const Query& query) const {
        return query.kind == "NEIGHBOR_TYPE_AGG" || query.kind == "TEMPORAL_TYPE_AGG"
            || query.kind == "TWO_HOP_REACHABILITY" || query.kind == "TWO_HOP_JOIN";
    }
    void loadMaterialized(const Query&) { selected.clear(); }
    std::string prefix(const std::string& verb) const {
        return verb + " " + std::to_string(queryIndex) + " " + std::to_string(worker) + " ";
    }
    void enqueue(int destination, const std::string& message, bool retry = false) {
        protocolOps.add(Op::Control);
        if (message.rfind("DATA ",0) == 0) {
            const uint64_t offset = outboundSpool.append(message);
            outbound.push_back({destination, "", retry, offset, true});
        } else outbound.push_back({destination, message, retry, 0, false});
        if (!sendEvent->isScheduled()) scheduleAt(simTime() + 1.0 / par("codecCpuUnitsPerSecond").doubleValue(), sendEvent);
    }
    void broadcast(const std::string& message) {
        for (int w = 0; w < workers; ++w) enqueue(w, message);
    }
    void transmit() {
        auto item = outbound.front(); outbound.pop_front();
        if (item.spooled) item.payload = outboundSpool.at(item.spoolOffset);
        const std::string dropVerb = par("dropFirstVerb").stdstringValue();
        if (!injectedDrop && queryIndex == par("dropQueryIndex").intValue() && !dropVerb.empty() && item.payload.rfind(dropVerb + " ", 0) == 0) {
            injectedDrop = true;
            recordScalar("injectedLostMessage", 1);
            if (!outbound.empty()) scheduleAt(simTime() + 0.000005 + 1.0 / par("codecCpuUnitsPerSecond").doubleValue(), sendEvent);
            return;
        }
        auto *packet = new Packet(item.retry ? "graph-query-retry" : "graph-query");
        std::vector<uint8_t> bytes(item.payload.begin(), item.payload.end());
        packet->insertAtBack(makeShared<BytesChunk>(bytes));
        std::string host = item.destination < 0 ? "coordinator" : "worker[" + std::to_string(item.destination) + "]";
        sentBytes += bytes.size();
        socket.sendTo(packet, L3AddressResolver().resolve(host.c_str()), 9100);
        if (!outbound.empty()) scheduleAt(simTime() + 0.000005 + 1.0 / par("codecCpuUnitsPerSecond").doubleValue(), sendEvent);
    }
    void beginQuery() {
        if (++queryIndex == (int)queries.size()) { output.flush(); endSimulation(); return; }
        version = 0;
        plan = par("initialPlan");
        auto registeredInitial = initialPlans.find(queries[queryIndex].id);
        if (registeredInitial != initialPlans.end()) plan = registeredInitial->second;
        phase = 1; retries = 0; decisionRetries = 0; cancelled = false;
        rowCounts.clear(); distinctCounts.clear(); labelDigitSums.clear(); scannedEdgeCounts.clear();
        serviceTimes.clear(); scanCpuTimes.clear(); scanIoTimes.clear(); scanQueueTimes.clear(); workerCpuRates.clear(); acknowledgements.clear();
        rejected.clear(); chunks.clear(); expectedChunks.clear(); receivedChunks.clear(); completeWorkers.clear(); result.clear();
        decisionReason = "static_locked"; predictedCurrent = predictedBest = 0; switchWireBytes = 0;
        decisionLatencyNs = 0;
        scanOps.clear(); encodeOps.clear(); decodeOps.clear(); mergeOps.clear(); decisionOps.clear(); protocolOps.clear();
        decisionCpuService = encodeCpuService = decodeCpuService = 0;
        coordinatorQueueWait = 0; coordinatorCpuAvailable = simTime();
        result.configure(spoolPath("result"), par("groupingMemoryBytes").intValue());
        outboundSpool.configure(spoolPath("outbound"));
        receivedSpool.configure(spoolPath("received"));
        startRetryWireBytes = retryWireBytes;
        resultPayloadBytes = 0;
        started = simTime(); startSent = sentBytes; startReceived = receivedBytes; startWire = wireBytes;
        broadcast("SCAN " + std::to_string(queryIndex) + " -1");
        scheduleAt(simTime() + par("timeout"), deadlineEvent);
        scheduleAt(simTime() + par("retryInterval"), retryEvent);
        if (queryIndex == 0 && par("cancelAfter").doubleValue() >= 0)
            scheduleAt(simTime() + par("cancelAfter"), cancelQueryEvent);
    }
    void cancelQuery() {
        if (phase == 4) return;
        cancelEvent(decisionEvent);
        phase = 5; cancelled = true; barrierAt = simTime(); acknowledgements.clear();
        broadcast("CANCEL " + std::to_string(queryIndex) + " -1");
    }
    void retryMissing() {
        if (phase == 4) return;
        if (++retries > par("maxRetries").intValue())
            throw cRuntimeError("QUERY_RETRY_EXHAUSTED: no successful result");
        for (int w = 0; w < workers; ++w) {
            if (phase == 1 && !rowCounts.count(w))
                enqueue(w, "SCAN " + std::to_string(queryIndex) + " -1", true);
            if (phase == 2 && !acknowledgements.count(w))
                enqueue(w, "PREPARE " + std::to_string(queryIndex) + " -1 " + std::to_string(proposed), true);
            if (phase == 3 && !completeWorkers.count(w))
                enqueue(w, "EXECUTE " + std::to_string(queryIndex) + " -1 " + std::to_string(version) + " " + std::to_string(plan), true);
            if (phase == 5 && !acknowledgements.count(w))
                enqueue(w, "CANCEL " + std::to_string(queryIndex) + " -1", true);
        }
        scheduleAt(simTime() + par("retryInterval"), retryEvent);
    }
    void scan(int index) {
        if (index < queryIndex) return; // delayed request cannot roll a worker back
        if (index == queryIndex && !cachedReady.empty()) {
            enqueue(-1, cachedReady, true); return;
        }
        if (index == queryIndex) return; // duplicate request while service in flight
        queryIndex = index; version = 0; selected.clear(); grouped.clear(); cachedReply.clear(); cachedReady.clear();
        recordScalar("scanExecution", 1);
        const auto& q = queries.at(index);
        if (isMaterializedQuery(q)) {
            // Every safe-point input is an immutable query artifact carrying
            // its own graph-version checksum. Query order therefore need not be
            // monotonic across query classes or parameter schedules.
            appliedVersion = q.graphVersion;
            loadMaterialized(q);
        } else {
            applyUpdates(q.graphVersion);
            selectProjection(edges, q, selected);
        }
        scanOps.clear(); encodeOps.clear(); protocolOps.clear();
        encodeCpuService = 0;
        grouped.configure(spoolPath("groups"), par("groupingMemoryBytes").intValue());
        outboundSpool.configure(spoolPath("outbound"));
        selectedCount = committedLabelDigits = 0;
        visitSelected([&](uint64_t token) {
            ++selectedCount; committedLabelDigits += std::to_string(token).size();
            scanOps.add(Op::Tuple); scanOps.add(Op::Aggregate); grouped.add(token);
        });
        grouped.finalize();
        localScannedRows = isMaterializedQuery(q) ? partitionSum(q.scanRowsByPartition) : edges.size();
        scanOps.add(Op::EdgeVisit, localScannedRows);
        scanOps.add(Op::Predicate, localScannedRows);
        const uint64_t base = isMaterializedQuery(q) ? partitionSum(q.scanWorkByPartition) : 2 * edges.size();
        if (base > 2 * localScannedRows) scanOps.add(Op::Join, base - 2 * localScannedRows);
        committedChecksum = checksum();
        spillBytes = std::max(0.0, selectedCount * 8.0 - par("memoryBytes").doubleValue());
        const double effectiveCpuRate = par("scanCpuUnitsPerSecond").doubleValue() * par("cpuMultiplier").doubleValue();
        const double scanWork = scanOps.total();
        scanCpuService = scanWork / effectiveCpuRate;
        scanIoService = (localScannedRows * 28.0 + selectedCount * 8.0 + spillBytes
            + grouped.readBytes + grouped.writtenBytes) / par("ioBytesPerSecond").doubleValue();
        scanQueueService = par("queueDelay").doubleValue();
        scanService = scanCpuService + scanIoService + scanQueueService;
        phase = 1; scanStarted = simTime();
        scheduleAt(simTime() + scanService, serviceEvent);
    }
    void scanFinished() {
        std::ostringstream response;
        const auto& q = queries.at(queryIndex);
        const uint64_t labelDigits = committedLabelDigits;
        response << prefix("READY") << selectedCount << " " << grouped.size() << " "
                 << committedChecksum << " " << std::setprecision(17) << (simTime() - scanStarted).dbl()
                 << " " << labelDigits << " " << appliedVersion << " "
                 << localScannedRows
                 << " " << scanCpuService << " " << scanIoService << " " << scanQueueService
                 << " " << par("codecCpuUnitsPerSecond").doubleValue() * par("cpuMultiplier").doubleValue();
        cachedReady = response.str(); enqueue(-1, cachedReady);
        recordScalar("scanRows", (double)edges.size());
        recordScalar("selectedRows", (double)selectedCount);
        recordScalar("scanServiceSeconds", scanService);
        recordScalar("scanCpuServiceSeconds", scanCpuService);
        recordScalar("scanIoServiceSeconds", scanIoService);
        recordScalar("scanQueueServiceSeconds", scanQueueService);
        recordScalar("spillBytes", spillBytes);
        recordScalar("observedServiceUnitsPerSecond", edges.size() / std::max(1e-12, scanService));
    }
    void decidePlan() {
        decisionOps.clear();
        barrierAt = simTime();
        decisionRetries = retries;
        uint64_t rows = 0, distinct = 0, labelDigits = 0;
        for (auto item : rowCounts) { rows += item.second; decisionOps.add(Op::Arithmetic); }
        for (auto item : distinctCounts) { distinct += item.second; decisionOps.add(Op::Arithmetic); }
        for (auto item : labelDigitSums) { labelDigits += item.second; decisionOps.add(Op::Arithmetic); }
        const double meanLabel = rows ? decisionMath.div(labelDigits,rows) : 1.0;
        observedMeanLabel = meanLabel;
        const double networkHeader = par("networkHeaderBytes").doubleValue();
        const double mtu = par("networkMtuBytes").doubleValue();
        const double appChunk = par("applicationChunkBytes").doubleValue();
        if (networkHeader < 0 || mtu <= networkHeader || appChunk <= 0)
            throw cRuntimeError("invalid cost-model network framing");
        auto framedWorker = [&](double payload, bool applicationChunked) {
            const double fragmentPayload = applicationChunked
                ? decisionMath.min(appChunk, decisionMath.sub(mtu,networkHeader))
                : decisionMath.sub(mtu,networkHeader);
            const double fragments = decisionMath.max(1.0, decisionMath.ceil(decisionMath.div(payload,fragmentPayload)));
            return decisionMath.add(payload,decisionMath.mul(fragments,networkHeader));
        };
        // Use committed per-worker counts so packet framing remains valid under
        // skew. Candidate payloads are still not materialized before selection.
        double raw = 0, partial = 0, compressed = 0;
        const double meanRows = workers ? decisionMath.div(rows,workers) : 0.0;
        double squaredDeviation = 0;
        for (int workerId = 0; workerId < workers; ++workerId) {
            const double localRows = rowCounts.count(workerId) ? rowCounts.at(workerId) : 0.0;
            const double localDistinct = distinctCounts.count(workerId) ? distinctCounts.at(workerId) : 0.0;
            const double localLabelDigits = labelDigitSums.count(workerId) ? labelDigitSums.at(workerId) : 0.0;
            const double localMeanLabel = localRows ? decisionMath.div(localLabelDigits,localRows) : 1.0;
            const double localMeanCount = localDistinct ? decisionMath.div(localRows,localDistinct) : 1.0;
            const double localCountDigits = decisionMath.add(decisionMath.floor(
                decisionMath.log10(decisionMath.max(1.0, localMeanCount))), 1.0);
            const double localLabelVarint = decisionMath.max(1.0,
                decisionMath.ceil(decisionMath.div(decisionMath.mul(localMeanLabel,decisionMath.log2(10.0)),7.0)));
            const double localCountVarint = decisionMath.max(1.0,
                decisionMath.ceil(decisionMath.div(decisionMath.log2(decisionMath.max(2.0,decisionMath.add(localMeanCount,1.0))),7.0)));
            raw = decisionMath.add(raw,framedWorker(decisionMath.add(decisionMath.mul(localRows,decisionMath.add(localMeanLabel,1.0)),32.0),true));
            partial = decisionMath.add(partial,framedWorker(decisionMath.add(decisionMath.mul(localDistinct,decisionMath.add(decisionMath.add(localMeanLabel,localCountDigits),2.0)),32.0),true));
            compressed = decisionMath.add(compressed,framedWorker(decisionMath.add(decisionMath.mul(localDistinct,decisionMath.add(localLabelVarint,localCountVarint)),292.0),false));
            squaredDeviation = decisionMath.add(squaredDeviation,decisionMath.mul(decisionMath.sub(localRows,meanRows),decisionMath.sub(localRows,meanRows)));
        }
        observedPartitionSkewCv = meanRows > 0
            ? decisionMath.div(decisionMath.sqrt(decisionMath.div(squaredDeviation,workers)),meanRows) : 0.0;
        const std::vector<double> predicted = {raw, partial, compressed};
        predictedPlanBytes = predicted;
        predictedCurrent = predicted.at(plan);
        proposed = (int)std::distance(predicted.begin(), std::min_element(predicted.begin(), predicted.end(), [&](double a,double b) { decisionOps.add(Op::Comparison); return a < b; }));
        predictedBest = predicted.at(proposed);
        policyFallback = false;
        const double saving = decisionMath.sub(predictedCurrent,predictedBest);
        const double relativeStd = decisionMath.mul(par("predictionRelativeStd").doubleValue(),decisionMath.add(1.0,observedPartitionSkewCv));
        const double uncertaintyPenalty = decisionMath.mul(par("uncertaintyZ").doubleValue(),decisionMath.hypot(decisionMath.mul(predictedCurrent,relativeStd),decisionMath.mul(predictedBest,relativeStd)));
        const double requiredBenefit = decisionMath.add(decisionMath.add(estimatedSwitchWire,uncertaintyPenalty),decisionMath.mul(par("benefitSafetyMargin").doubleValue(),predictedCurrent));
        decisionOps.add(Op::Comparison);
        if (method == "STATIC") { proposed = plan; decisionReason = "static_locked"; }
        else if (method == "SWITCH_COST_THRESHOLD") {
            if (!switchThresholdLoaded)
                throw cRuntimeError("switch-threshold artifact was not loaded");
            const double required = decisionMath.add(estimatedSwitchWire,decisionMath.mul(switchThreshold,predictedCurrent));
            if (decisionMath.le(saving,required)) {
                proposed = plan;
                decisionReason = "below_switch_cost_threshold";
            } else decisionReason = proposed == plan
                ? "switch_cost_current_best" : "switch_cost_threshold_switch";
        } else if (method == "SAFE_BENEFIT" && decisionMath.le(saving,requiredBenefit)) {
            proposed = plan; decisionReason = "insufficient_net_transfer_saving";
        } else if (method == "LINUCB") {
            const double compressionRatio = decisionMath.div(compressed,decisionMath.max(1.0,partial));
            const auto context = modelContext(rows, distinct, meanLabel, compressionRatio);
            std::vector<double> model(context.size(), 0.0);
            if (policyLoaded) for (size_t i = 0; i < context.size(); ++i) {
                decisionOps.add(Op::Arithmetic, 2);
                model[i] = (context[i] - policyMean[i]) / policyScale[i];
            }
            double oodSquared = 0;
            if (policyLoaded) for (size_t row = 0; row < model.size(); ++row)
                for (size_t column = 0; column < model.size(); ++column) {
                    decisionOps.add(Op::Arithmetic, 3);
                    oodSquared += model[row]
                        * policyOodInverse[row * model.size() + column]
                        * model[column];
                }
            const double oodDistance = decisionMath.sqrt(decisionMath.max(0.0, oodSquared));
            if (!policyLoaded || policyForceFallback || decisionMath.greater(oodDistance,oodThreshold)) {
                policyFallback = true;
                if (decisionMath.le(saving,requiredBenefit)) proposed = plan;
                if (!policyLoaded) decisionReason = "rl_missing_fallback_SAFE_BENEFIT";
                else if (policyForceFallback) decisionReason = "rl_validation_fallback_SAFE_BENEFIT";
                else decisionReason = "rl_ood_fallback_SAFE_BENEFIT";
            } else {
                const auto scores = policyScores(model);
                proposed = (int)std::distance(scores.begin(), std::max_element(scores.begin(), scores.end(), [&](double a,double b) { decisionOps.add(Op::Comparison); return a < b; }));
                decisionReason = proposed == plan ? "rl_keep" : "rl_switch";
            }
        } else decisionReason = proposed == plan ? "current_best" : "positive_net_transfer_saving";
        decisionCpuService = decisionOps.total() / par("codecCpuUnitsPerSecond").doubleValue();
        decisionLatencyNs = static_cast<uint64_t>(std::llround(decisionCpuService * 1e9));
        phase = 7;
        scheduleAt(simTime() + decisionCpuService, decisionEvent);
    }
    void commitDecision() {
        acknowledgements.clear(); rejected.clear();
        if (proposed == plan && !par("alwaysPrepare").boolValue()) {
            phase = 3;
            broadcast("EXECUTE " + std::to_string(queryIndex) + " -1 " + std::to_string(version) + " " + std::to_string(plan));
        } else {
            // PREPARE/ACK is normally an incremental switching cost. Fault tests may force
            // this barrier without changing the committed plan to exercise ACK handling.
            phase = 2;
            startSwitchWire = wireBytes;
            broadcast("PREPARE " + std::to_string(queryIndex) + " -1 " + std::to_string(proposed));
        }
    }
    void buildReplies(int selectedPlan, int selectedVersion) {
        MessageSpool payloads; payloads.configure(spoolPath("payloads"));
        std::string chunk;
        auto emit = [&](const std::string& bytes) {
            encodeOps.add(Op::EncodeByte, bytes.size()); payloads.append(bytes);
        };
        auto append = [&](const std::string& token) {
            if (chunk.size() + token.size() > 900 && !chunk.empty()) { emit(chunk); chunk.clear(); }
            chunk += token;
        };
        if (selectedPlan == 0) visitSelected([&](uint64_t label) {
            encodeOps.add(Op::Tuple); append(std::to_string(label) + " ");
        });
        else if (selectedPlan == 1) grouped.forEach([&](uint64_t key,uint64_t count) {
            encodeOps.add(Op::Tuple); append(std::to_string(key) + ":" + std::to_string(count) + " ");
        });
        else if (selectedPlan == 2) {
            const std::string path = spoolPath("compressed");
            {
                std::ofstream encoded(path,std::ios::binary | std::ios::trunc);
                std::string header(256,'\0'); header.replace(0,4,"SBC1",4); encoded.write(header.data(),header.size());
                const uint32_t entries = grouped.size();
                for (int i=0;i<4;++i) encoded.put(static_cast<char>((entries >> (8*i)) & 255));
                uint64_t previous = 0;
                grouped.forEach([&](uint64_t key,uint64_t count) {
                    std::string pair; appendVarint(pair,key-previous); appendVarint(pair,count); previous=key;
                    encodeOps.add(Op::Tuple); encodeOps.add(Op::Arithmetic);
                    encoded.write(pair.data(),pair.size());
                });
                if (!encoded) throw cRuntimeError("cannot spool compressed result");
            }
            std::ifstream encoded(path,std::ios::binary | std::ios::ate);
            const uint64_t length = static_cast<uint64_t>(encoded.tellg());
            encoded.seekg(0);
            constexpr size_t maxUdpApplicationBytes = 65507;
            constexpr size_t largeCompressedChunkBytes = 60000;
            const std::string singleHeader = prefix("DATA") + std::to_string(selectedVersion) + " 0 1 ";
            const size_t block = length + singleHeader.size() <= maxUdpApplicationBytes ? length : largeCompressedChunkBytes;
            std::string bytes(std::max<size_t>(1,block),'\0');
            while (encoded.read(bytes.data(),bytes.size()) || encoded.gcount()) emit(std::string(bytes.data(),encoded.gcount()));
            encoded.close(); std::filesystem::remove(path);
        } else throw cRuntimeError("unknown physical plan");
        if (!chunk.empty() || payloads.size()==0) emit(chunk);
        cachedReply.configure(spoolPath("replies"));
        size_t index=0;
        payloads.forEach([&](const std::string& payload) {
            cachedReply.append(prefix("DATA") + std::to_string(selectedVersion) + " "
                + std::to_string(index++) + " " + std::to_string(payloads.size()) + " " + payload);
        });
    }
    void executeSuffix() {
        if (checksum() != committedChecksum) throw cRuntimeError("materialization corruption");
        recordScalar("suffixExecution", 1); encodeOps.clear();
        const uint64_t beforeRead = grouped.readBytes;
        buildReplies(plan,version); phase = 3;
        suffixIoService = (grouped.readBytes - beforeRead) / par("ioBytesPerSecond").doubleValue();
        encodeCpuService = encodeOps.total() / (par("codecCpuUnitsPerSecond").doubleValue()
            * par("cpuMultiplier").doubleValue());
        scheduleAt(simTime() + encodeCpuService + suffixIoService, serviceEvent);
    }
    void finishQuery() {
        cancelEvent(deadlineEvent);
        cancelEvent(retryEvent);
        cancelEvent(cancelQueryEvent);
        cancelEvent(decisionEvent);
        uint64_t selectedRows = 0, localDistinctGroups = 0, scannedEdges = 0;
        double scanWorkerSum = 0, scanCriticalPath = 0, scanCpuWorkerSum = 0, scanIoWorkerSum = 0, scanQueueWorkerSum = 0;
        double suffixCpuWorkerSum = 0, suffixCpuCriticalPath = 0;
        for (const auto& item : rowCounts) selectedRows += item.second;
        for (const auto& item : distinctCounts) localDistinctGroups += item.second;
        for (const auto& item : scannedEdgeCounts) scannedEdges += item.second;
        for (const auto& item : serviceTimes) { scanWorkerSum += item.second; scanCriticalPath = std::max(scanCriticalPath, item.second); }
        for (const auto& item : scanCpuTimes) scanCpuWorkerSum += item.second;
        for (const auto& item : scanIoTimes) scanIoWorkerSum += item.second;
        for (const auto& item : scanQueueTimes) scanQueueWorkerSum += item.second;
        OperationCounts allScan, allEncode, allProtocol;
        uint64_t groupingRead = result.readBytes, groupingWrite = result.writtenBytes, peakBuffer = result.peakBufferBytes;
        for (int index=0; index<workers; ++index) {
            const auto* app = workerApp(index);
            allScan.accumulate(app->scanOps); allEncode.accumulate(app->encodeOps);
            allProtocol.accumulate(app->protocolOps);
            const double suffix = app->encodeCpuService;
            suffixCpuWorkerSum += suffix; suffixCpuCriticalPath = std::max(suffixCpuCriticalPath,suffix);
            groupingRead += app->grouped.readBytes; groupingWrite += app->grouped.writtenBytes;
            peakBuffer = std::max<uint64_t>(peakBuffer,app->grouped.peakBufferBytes);
        }
        allProtocol.accumulate(protocolOps);
        const uint64_t mergeCpuWorkUnits = mergeOps.total();
        const uint64_t suffixCpuWorkUnits = allEncode.total();
        const double coordinatorMergeCpuService = mergeOps.total() / par("codecCpuUnitsPerSecond").doubleValue();
        const double protocolCpuService = allProtocol.total() / par("codecCpuUnitsPerSecond").doubleValue();
        const double totalCpuService = scanCpuWorkerSum + suffixCpuWorkerSum
            + decodeCpuService + coordinatorMergeCpuService + decisionCpuService + protocolCpuService;
        const uint64_t queryWireBytes = wireBytes - startWire;
        const uint64_t queryRetryWireBytes = std::min(
            queryWireBytes, retryWireBytes - startRetryWireBytes);
        uint64_t remainingWireBytes = queryWireBytes - queryRetryWireBytes;
        const uint64_t payloadWireBytes = std::min(remainingWireBytes, resultPayloadBytes);
        remainingWireBytes -= payloadWireBytes;
        const uint64_t attributedSwitchWireBytes = std::min(remainingWireBytes, switchWireBytes);
        const uint64_t controlWireBytes = remainingWireBytes - attributedSwitchWireBytes;
        output << "{\"query_id\":\"" << queries[queryIndex].id << "\",\"query_class\":\"" << queries[queryIndex].kind
            << "\",\"method\":\"" << method << "\",\"workers\":" << workers << ",\"graph_version\":" << queries[queryIndex].graphVersion << ",\"plan_version\":" << version
            << ",\"initial_plan\":" << (initialPlans.count(queries[queryIndex].id) ? initialPlans.at(queries[queryIndex].id) : par("initialPlan").intValue())
            << ",\"plan\":" << plan << ",\"decision\":\"" << decisionReason << "\",\"latency_simtime_ms\":" << (simTime() - started).dbl() * 1000
            << ",\"decision_latency_ns\":" << decisionLatencyNs
            << ",\"barrier_and_suffix_ms\":" << (simTime() - barrierAt).dbl() * 1000
            << ",\"scanned_edges\":" << scannedEdges
            << ",\"graph_vertex_count\":" << queries[queryIndex].graphVertexCount
            << ",\"graph_edge_count\":" << queries[queryIndex].graphEdgeCount
            << ",\"network_intermediate_nodes\":1,\"network_hops\":2"
            << ",\"scan_service_worker_sum_ms\":" << scanWorkerSum * 1000
            << ",\"scan_service_critical_path_ms\":" << scanCriticalPath * 1000
            << ",\"scan_cpu_service_worker_sum_ms\":" << scanCpuWorkerSum * 1000
            << ",\"scan_io_service_worker_sum_ms\":" << scanIoWorkerSum * 1000
            << ",\"scan_queue_wait_worker_sum_ms\":" << scanQueueWorkerSum * 1000
            << ",\"worker_suffix_cpu_work_units\":" << suffixCpuWorkUnits
            << ",\"worker_suffix_cpu_service_sum_ms\":" << suffixCpuWorkerSum * 1000
            << ",\"worker_suffix_cpu_critical_path_ms\":" << suffixCpuCriticalPath * 1000
            << ",\"coordinator_merge_cpu_work_units\":" << mergeCpuWorkUnits
            << ",\"coordinator_merge_cpu_service_ms\":" << coordinatorMergeCpuService * 1000
            << ",\"total_modeled_cpu_service_ms\":" << totalCpuService * 1000
            << ",\"coordinator_sent_payload_bytes\":" << sentBytes - startSent << ",\"coordinator_received_payload_bytes\":" << receivedBytes - startReceived
            << ",\"total_transfer_payload_bytes\":" << (sentBytes - startSent) + (receivedBytes - startReceived)
            << ",\"payload_bytes\":" << payloadWireBytes
            << ",\"control_bytes\":" << controlWireBytes
            << ",\"retry_bytes\":" << queryRetryWireBytes
            << ",\"total_wire_bytes\":" << queryWireBytes
            << ",\"switch_wire_bytes\":" << attributedSwitchWireBytes
            << ",\"predicted_current_transfer_bytes\":" << predictedCurrent << ",\"predicted_best_transfer_bytes\":" << predictedBest
            << ",\"predicted_plan_wire_bytes\":{\"0\":" << predictedPlanBytes.at(0)
            << ",\"1\":" << predictedPlanBytes.at(1)
            << ",\"2\":" << predictedPlanBytes.at(2) << "}"
            << ",\"partition_skew_cv\":" << observedPartitionSkewCv
            << ",\"estimated_switch_wire_bytes\":" << estimatedSwitchWire
            << ",\"selected_rows\":" << selectedRows
            << ",\"distinct_groups\":" << localDistinctGroups
            << ",\"mean_label_bytes\":" << observedMeanLabel
            << ",\"policy_fallback\":" << (policyFallback ? "true" : "false")
            << ",\"decision_retry_count\":" << decisionRetries
            << ",\"decision_safe_point\":true,\"switch_budget_before_decision\":1"
            << ",\"retry_rounds\":" << retries
            << ",\"outcome\":\"" << (cancelled ? "CANCELLED" : "SUCCESS") << "\""
            << ",\"decision_cpu_service_ms\":" << decisionCpuService*1000
            << ",\"decode_cpu_service_ms\":" << decodeCpuService*1000
            << ",\"protocol_cpu_service_ms\":" << protocolCpuService*1000
            << ",\"coordinator_queue_wait_ms\":" << coordinatorQueueWait*1000
            << ",\"modeled_io_read_bytes\":" << scannedEdges*28 + selectedRows*8 + groupingRead
            << ",\"modeled_io_write_bytes\":" << groupingWrite
            << ",\"modeled_spill_bytes\":" << groupingWrite
            << ",\"grouping_io_read_bytes\":" << groupingRead
            << ",\"grouping_io_write_bytes\":" << groupingWrite
            << ",\"peak_group_buffer_estimated_bytes\":" << peakBuffer
            << ",\"reference_model_id\":\"logical-ops-reference-2\""
            << ",\"operation_counts_by_phase\":{\"scan\":";
        allScan.json(output); output << ",\"encode\":"; allEncode.json(output);
        output << ",\"decode\":"; decodeOps.json(output);
        output << ",\"merge\":"; mergeOps.json(output);
        output << ",\"decision\":"; decisionOps.json(output);
        output << ",\"protocol\":"; allProtocol.json(output);
        output << "},\"result\":{";
        bool first = true;
        result.finalize();
        result.forEach([&](uint64_t key,uint64_t count) {
            if (!first) output << ","; first = false; output << "\"" << key << "\":" << count;
        });
        output << "}}\n"; output.flush(); phase = 4;
        scheduleAt(simTime() + .01, startEvent);
    }
  protected:
    int numInitStages() const override { return NUM_INIT_STAGES; }
    void initialize(int stage) override {
        if (stage == INITSTAGE_LOCAL) {
            coordinator = par("coordinator"); worker = par("workerId"); workers = par("workerCount"); method = par("method").stdstringValue();
            if (workers != 1 && workers != 2 && workers != 4 && workers != 8 && workers != 16)
                throw cRuntimeError("supported worker counts: 1,2,4,8,16");
            if (method != "STATIC" && method != "SWITCH_COST_THRESHOLD"
                && method != "LINUCB" && method != "SAFE_BENEFIT")
                throw cRuntimeError("unsupported transfer method");
            if (par("scanCpuUnitsPerSecond").doubleValue() <= 0
                || par("codecCpuUnitsPerSecond").doubleValue() <= 0
                || par("cpuMultiplier").doubleValue() <= 0
                || par("ioBytesPerSecond").doubleValue() <= 0)
                throw cRuntimeError("invalid service capacity");
            if (par("retryInterval").doubleValue() <= 0 || par("maxRetries").intValue() < 0)
                throw cRuntimeError("invalid retry policy");
            startEvent = new cMessage("next-query"); serviceEvent = new cMessage("service"); sendEvent = new cMessage("send"); deadlineEvent = new cMessage("deadline");
            retryEvent = new cMessage("retry-missing");
            cancelQueryEvent = new cMessage("cancel-query");
            decisionEvent = new cMessage("decision-service");
            std::string directory = par("inputDirectory").stdstringValue();
            std::string queryPath = par("queryFile").stdstringValue();
            if (queryPath.empty()) queryPath = directory + "/queries.tsv";
            std::ifstream queryFile(queryPath);
            std::string queryLine;
            while (std::getline(queryFile, queryLine)) {
                if (queryLine.empty()) continue;
                std::istringstream queryStream(queryLine);
                Query q;
                if (!(queryStream >> q.id >> q.kind >> q.seed >> q.begin >> q.end
                    >> q.graphVersion >> q.groupModulus >> q.samplingDenominator))
                    throw cRuntimeError("malformed query row");
                if (!(queryStream >> q.graphVertexCount >> q.graphEdgeCount >> q.scanRows)) {
                    q.graphVertexCount = q.graphEdgeCount = q.scanRows = 0;
                    queryStream.clear();
                }
                std::string scanWork, scanRows;
                if (queryStream >> scanWork >> scanRows) {
                    auto numbers = [](const std::string& field) {
                        std::vector<uint64_t> out; std::stringstream in(field); std::string value;
                        while (std::getline(in,value,',')) out.push_back(std::stoull(value));
                        return out;
                    };
                    q.scanWorkByPartition = numbers(scanWork); q.scanRowsByPartition = numbers(scanRows);
                }
                if (isMaterializedQuery(q) && (q.scanWorkByPartition.size() != 8 || q.scanRowsByPartition.size() != 8))
                    throw cRuntimeError("materialized query requires reference-model operation metadata");
                if (q.groupModulus == 0 || q.samplingDenominator == 0)
                    throw cRuntimeError("invalid zero query grouping/sampling value");
                queries.push_back(q);
            }
            if (queries.empty()) throw cRuntimeError("missing query input");
            std::string initialPlanPath = par("initialPlanFile").stdstringValue();
            if (!initialPlanPath.empty()) {
                std::ifstream plans(initialPlanPath);
                std::string id; int registeredPlan;
                while (plans >> id >> registeredPlan) {
                    if (registeredPlan < 0 || registeredPlan > 2 || initialPlans.count(id))
                        throw cRuntimeError("invalid registered initial plan");
                    initialPlans[id] = registeredPlan;
                }
                if (!plans.eof()) throw cRuntimeError("malformed initial-plan file");
            }
            const bool usesMaterialized = !par("materializedDirectory").stdstringValue().empty();
            if (!coordinator && !usesMaterialized) for (int part = 0; part < par("partitionCount").intValue(); ++part) if (part % workers == worker) {
                std::ifstream input(directory + "/part-" + std::to_string(part) + ".bin", std::ios::binary);
                if (!input) throw cRuntimeError("missing graph partition");
                EdgeRow row; while (input.read(reinterpret_cast<char*>(&row), sizeof(row))) edges.push_back(row);
                if (input.gcount() != 0) throw cRuntimeError("truncated partition record");
            }
            if (!coordinator && !usesMaterialized) {
                std::string updateDirectory = par("updateDirectory").stdstringValue();
                if (!updateDirectory.empty()) {
                    for (int part = 0; part < par("partitionCount").intValue(); ++part) if (part % workers == worker) {
                        std::ifstream input(updateDirectory + "/part-" + std::to_string(part) + ".bin", std::ios::binary);
                        if (!input) throw cRuntimeError("missing update partition");
                        UpdateRow row; while (input.read(reinterpret_cast<char*>(&row), sizeof(row))) updates.push_back(row);
                        if (input.gcount() != 0) throw cRuntimeError("truncated update record");
                    }
                    std::sort(updates.begin(), updates.end(), [](const UpdateRow& a, const UpdateRow& b) {
                        return a.sequence < b.sequence;
                    });
                }
            }
            if (coordinator) {
                estimatedSwitchWire = par("estimatedSwitchWireBytes").doubleValue();
                if (estimatedSwitchWire < 0) throw cRuntimeError("negative estimated switch cost");
                loadPolicy(par("policyFile").stdstringValue());
                loadSwitchThreshold(par("switchThresholdFile").stdstringValue());
                output.open(par("outputFile").stringValue()); if (!output) throw cRuntimeError("cannot open output"); output << std::setprecision(17);
            }
        }
        if (stage == INITSTAGE_APPLICATION_LAYER) {
            socket.setOutputGate(gate("socketOut")); socket.bind(9100); socket.setCallback(this);
            if (coordinator) {
                cModule *network = getParentModule()->getParentModule();
                std::vector<cModule *> hosts = {network->getSubmodule("coordinator")};
                for (int i = 0; i < workers; ++i) hosts.push_back(network->getSubmodule("worker", i));
                for (auto *host : hosts) {
                    cModule *interface = host ? host->getSubmodule("eth", 0) : nullptr;
                    cModule *mac = interface ? interface->getSubmodule("mac") : nullptr;
                    if (!mac) throw cRuntimeError("cannot locate host Ethernet MAC for wire accounting");
                    mac->subscribe("packetSentToLower", this);
                    wireSources.push_back(mac);
                }
                scheduleAt(.2, startEvent);
            }
        }
    }
    void handleMessage(cMessage *msg) override {
        if (msg == decisionEvent) commitDecision();
        else if (msg == sendEvent) transmit();
        else if (msg == retryEvent) retryMissing();
        else if (msg == cancelQueryEvent) cancelQuery();
        else if (msg == startEvent) beginQuery();
        else if (msg == deadlineEvent) throw cRuntimeError("QUERY_TIMEOUT: incomplete protocol, not a successful result");
        else if (msg == serviceEvent) {
            if (coordinator) finishQuery();
            else if (phase == 1) scanFinished();
            else cachedReply.forEach([&](const std::string& text) { enqueue(-1, text); if (par("duplicateReplies")) { recordScalar("duplicateReplyGenerated",1); enqueue(-1, text); } });
        } else socket.processMessage(msg);
    }
    void socketDataArrived(UdpSocket *, Packet *packet) override {
        auto bytes = packet->peekDataAsBytes()->getBytes(); receivedBytes += bytes.size(); delete packet;
        std::istringstream message(std::string(bytes.begin(), bytes.end()));
        std::string verb; int index, sender;
        if (!(message >> verb >> index >> sender)) throw cRuntimeError("malformed message header");
        if (index < 0 || index >= (int)queries.size()) throw cRuntimeError("invalid query ID");
        if (!coordinator) {
            if (par("permanentFailure").boolValue()) return;
            if (verb == "CANCEL" && index >= queryIndex) {
                queryIndex = index; phase = 6;
                cancelEvent(serviceEvent); cancelEvent(sendEvent); outbound.clear();
                selected.clear(); grouped.clear(); cachedReply.clear(); cachedReady.clear();
                recordScalar("cancelAcknowledged",1); enqueue(-1, prefix("CANCELLED")); return;
            }
            if (phase == 6 && index == queryIndex) return;
            if (verb == "SCAN") { scan(index); return; }
            if (index != queryIndex) return;
            if (verb == "PREPARE") { message >> proposed; if (par("rejectPrepare").boolValue()) recordScalar("prepareRejected",1); enqueue(-1, prefix("ACK") + (par("rejectPrepare").boolValue() ? "0" : "1")); }
            else if (verb == "EXECUTE") {
                int requestedVersion, requestedPlan; message >> requestedVersion >> requestedPlan;
                if (!message || requestedPlan < 0 || requestedPlan > 2 || requestedVersion < 0)
                    throw cRuntimeError("invalid EXECUTE contract");
                if (phase == 3) {
                    if (requestedVersion != version || requestedPlan != plan)
                        throw cRuntimeError("retry attempted to change committed suffix");
                    if (!serviceEvent->isScheduled() && outbound.empty()) cachedReply.forEach([&](const std::string& text) { enqueue(-1, text, true); });
                    return;
                }
                if (cachedReady.empty()) throw cRuntimeError("EXECUTE before materialization");
                version = requestedVersion; plan = requestedPlan; executeSuffix();
            }
            return;
        }
        if (index != queryIndex || sender < 0 || sender >= workers) return;
        if (verb == "CANCELLED" && phase == 5) {
            acknowledgements.insert(sender);
            if ((int)acknowledgements.size() == workers) {
                result.clear(); decisionReason = "cancelled_all_workers_acknowledged"; finishQuery();
            }
        } else if (verb == "READY" && phase == 1) {
            uint64_t count, distinct, hash, labelDigits, workerVersion, scannedEdges;
            double service, scanCpu, scanIo, scanQueue, cpuRate;
            message >> count >> distinct >> hash >> service >> labelDigits >> workerVersion
                    >> scannedEdges >> scanCpu >> scanIo >> scanQueue >> cpuRate;
            if (!message) throw cRuntimeError("malformed READY payload");
            if (cpuRate <= 0 || scanCpu < 0 || scanIo < 0 || scanQueue < 0)
                throw cRuntimeError("invalid worker performance telemetry");
            if (workerVersion != queries[queryIndex].graphVersion) throw cRuntimeError("worker graph version mismatch");
            rowCounts[sender] = count; distinctCounts[sender] = distinct; serviceTimes[sender] = service;
            labelDigitSums[sender] = labelDigits; scannedEdgeCounts[sender] = scannedEdges;
            scanCpuTimes[sender] = scanCpu; scanIoTimes[sender] = scanIo;
            scanQueueTimes[sender] = scanQueue; workerCpuRates[sender] = cpuRate;
            if ((int)rowCounts.size() == workers) decidePlan();
        } else if (verb == "ACK" && phase == 2) {
            int accepted; message >> accepted; acknowledgements.insert(sender); if (!accepted) rejected.insert(sender);
            if ((int)acknowledgements.size() == workers) {
                if (rejected.empty() && proposed != plan) {
                    switchWireBytes = wireBytes - startSwitchWire;
                    plan = proposed; version++;
                }
                else if (!rejected.empty()) decisionReason = "fallback_prepare_failure";
                phase = 3;
                broadcast("EXECUTE " + std::to_string(queryIndex) + " -1 " + std::to_string(version) + " " + std::to_string(plan));
            }
        } else if (verb == "DATA" && phase == 3) {
            const uint64_t beforeWork = decodeOps.total() + mergeOps.total();
            const uint64_t beforeIo = result.readBytes + result.writtenBytes;
            int v, chunk, count; message >> v >> chunk >> count;
            if (v != version || count < 1 || chunk < 0 || chunk >= count) throw cRuntimeError("invalid data version/fragment");
            // DATA may carry a binary COMPRESSED_REDUCE payload. std::getline()
            // is not safe here because an encoded varint can contain byte 0x0a,
            // which would silently truncate the payload at an embedded newline.
            // Consume the complete remaining stream and remove only the protocol
            // separator that follows the textual DATA header.
            std::string payload{
                std::istreambuf_iterator<char>{message},
                std::istreambuf_iterator<char>{}};
            if (!payload.empty() && payload[0] == ' ') payload.erase(0, 1);
            auto key = std::make_pair(sender, chunk);
            if (chunks.count(key)) { if (receivedSpool.at(chunks[key]) != payload) throw cRuntimeError("conflicting duplicate tuple batch"); return; }
            if (expectedChunks.count(sender) && expectedChunks[sender] != count) throw cRuntimeError("inconsistent fragment total");
            expectedChunks[sender] = count; chunks[key] = receivedSpool.append(payload);
            resultPayloadBytes += payload.size();
            decodeOps.add(Op::DecodeByte, payload.size());
            if (plan != 2) {
                std::istringstream values(payload); std::string token;
                while (values >> token) {
                    auto separator = token.find(':');
                    if (plan == 0) {
                        if (separator != std::string::npos) throw cRuntimeError("raw tuple contains aggregate count");
                        result.add(std::stoull(token), 1); decodeOps.add(Op::Tuple); mergeOps.add(Op::Merge);
                    } else if (plan == 1) {
                        if (separator == std::string::npos) throw cRuntimeError("aggregate tuple misses count");
                        result.add(std::stoull(token.substr(0, separator)), std::stoull(token.substr(separator + 1))); decodeOps.add(Op::Tuple); mergeOps.add(Op::Merge);
                    }
                }
            }
            int received = ++receivedChunks[sender];
            if (received == count) {
                if (plan == 2) {
                    const std::string path = spoolPath("received-") + std::to_string(sender);
                    {
                        std::ofstream encoded(path,std::ios::binary | std::ios::trunc);
                        for (int part=0;part<count;++part) {
                            auto fragment=chunks.find(std::make_pair(sender,part));
                            if (fragment==chunks.end()) throw cRuntimeError("missing compressed DATA fragment");
                            const std::string bytes=receivedSpool.at(fragment->second);
                            encoded.write(bytes.data(),bytes.size());
                        }
                        if (!encoded) throw cRuntimeError("cannot spool incoming compressed result");
                    }
                    {
                        std::ifstream encoded(path,std::ios::binary);
                        decodeCompressedStream(encoded,[&](uint64_t key,uint64_t count) {
                            result.add(key,count); decodeOps.add(Op::Tuple); mergeOps.add(Op::Merge);
                        });
                    }
                    std::filesystem::remove(path);
                }
                completeWorkers.insert(sender);
            }
            if ((int)completeWorkers.size() == workers) result.finalize();
            const uint64_t work = decodeOps.total() + mergeOps.total() - beforeWork;
            const uint64_t io = result.readBytes + result.writtenBytes - beforeIo;
            coordinatorQueueWait += std::max(0.0,(coordinatorCpuAvailable - simTime()).dbl());
            coordinatorCpuAvailable = std::max(coordinatorCpuAvailable,simTime())
                + work / par("codecCpuUnitsPerSecond").doubleValue()
                + io / par("ioBytesPerSecond").doubleValue();
            if ((int)completeWorkers.size() == workers) {
                decodeCpuService = decodeOps.total() / par("codecCpuUnitsPerSecond").doubleValue();
                phase = 4; scheduleAt(coordinatorCpuAvailable,serviceEvent);
            }
        }
    }
    void socketErrorArrived(UdpSocket *, Indication *indication) override { delete indication; }
    void socketClosed(UdpSocket *) override {}
    void receiveSignal(cComponent *, simsignal_t, cObject *value, cObject *) override {
        auto *packet = dynamic_cast<cPacket *>(value);
        if (!packet || packet->getBitLength() < 0 || packet->getBitLength() % 8 != 0)
            throw cRuntimeError("invalid wire packet signal");
        const uint64_t bytes = packet->getBitLength() / 8;
        wireBytes += bytes;
        if (std::string(packet->getName()).find("graph-query-retry") != std::string::npos)
            retryWireBytes += bytes;
    }
    void finish() override {
        recordScalar("sentPayloadBytes", sentBytes);
        recordScalar("receivedPayloadBytes", receivedBytes);
        for (auto *source : wireSources) if (source) source->unsubscribe("packetSentToLower", this);
        wireSources.clear();
    }
  public:
    ~GraphQueryApp() override {
        cancelAndDelete(startEvent); cancelAndDelete(serviceEvent); cancelAndDelete(sendEvent); cancelAndDelete(deadlineEvent); cancelAndDelete(retryEvent); cancelAndDelete(cancelQueryEvent); cancelAndDelete(decisionEvent);
    }
};
Define_Module(GraphQueryApp);
