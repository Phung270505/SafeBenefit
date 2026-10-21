import inet.node.inet.StandardHost;
import inet.node.ethernet.EthernetSwitch;
import inet.networklayer.configurator.ipv4.Ipv4NetworkConfigurator;
import inet.common.scenario.ScenarioManager;
import inet.applications.contract.IApp;

simple GraphQueryApp like IApp
{
    parameters:
        bool coordinator = default(false);
        int workerId = default(-1);
        int workerCount = default(4);
        int partitionCount = default(8);
        string inputDirectory;
        string materializedDirectory = default("");
        string queryFile = default("");
        string updateDirectory = default("");
        string policyFile = default("");
        string switchThresholdFile = default("");
        string initialPlanFile = default("");
        string outputFile = default("");
        string spoolDirectory = default("");
        int groupingMemoryBytes = default(67108864);
        string method = default("STATIC");
        int initialPlan = default(0);
        double scanCpuUnitsPerSecond = default(1e7);
        double codecCpuUnitsPerSecond = default(1e7);
        double cpuMultiplier = default(1);
        double ioBytesPerSecond = default(1e9);
        double memoryBytes = default(1e9);
        double queueDelay @unit(s) = default(0s);
        double timeout @unit(s) = default(30s);
        bool rejectPrepare = default(false);
        bool duplicateReplies = default(false);
        double retryInterval @unit(s) = default(5s);
        int maxRetries = default(100);
        double benefitSafetyMargin = default(0.10);
        double estimatedSwitchWireBytes = default(512);
        double predictionRelativeStd = default(0.05);
        double uncertaintyZ = default(1.96);
        double networkHeaderBytes = default(42);
        double networkMtuBytes = default(1500);
        double applicationChunkBytes = default(900);
        bool alwaysPrepare = default(false);
        string dropFirstVerb = default("");
        int dropQueryIndex = default(0);
        bool permanentFailure = default(false);
        double cancelAfter @unit(s) = default(-1s);
        @class(GraphQueryApp);
    gates:
        input socketIn;
        output socketOut;
}

network GraphQuery
{
    parameters:
        int workers = default(4);
        double linkDatarate @unit(bps) = default(1Gbps);
        double linkDelay @unit(s) = default(50us);
    types:
        channel Link extends ned.DatarateChannel {
            datarate = parent.linkDatarate;
            delay = parent.linkDelay;
        }
    submodules:
        configurator: Ipv4NetworkConfigurator;
        scenarioManager: ScenarioManager;
        coordinator: StandardHost { parameters: numEthInterfaces = 1; }
        worker[workers]: StandardHost { parameters: numEthInterfaces = 1; }
        switch: EthernetSwitch { parameters: numEthInterfaces = parent.workers + 1; }
    connections:
        coordinator.ethg[0] <--> Link <--> switch.ethg[0];
        for i=0..workers-1 {
            worker[i].ethg[0] <--> Link <--> switch.ethg[i+1];
        }
}
