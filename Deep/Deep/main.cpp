#include "pch.h"
#include "DataProcessing/DataProcessing.h"
#include "Network/NetworkPrototype.h"
#include "NeuralNetworkCustomVariables.h"





#define traningCount 50000

#define clean false



int main() {

#if !clean


	using namespace TNNT;
	Timer t;

	//DataFormating start
	t.Start();


	DataSet data;
	{
		constexpr unsigned labelSize = 10;
		constexpr unsigned inputSize = 28 * 28;

		data.TrainingCount = traningCount;
		data.TrainingInputs = new float[data.TrainingCount * inputSize];
		data.TraningTargets = new float[data.TrainingCount * labelSize];

		data.ValidationCount = 10000;
		data.ValidationInputs = new float[data.ValidationCount * inputSize];
		data.ValidationTargets = new float[data.ValidationCount * labelSize];

		data.TestCount = 10000;
		data.TestInputs = new float[data.TestCount * inputSize];
		data.TestTargets = new float[data.TestCount * labelSize];

		//Data Formating start

		ProcessMNISTDataMT(10, data.TrainingInputs, data.TraningTargets, "../../TestAndTraningSets/trainLabel.idx1-ubyte", "../../TestAndTraningSets/trainIm.idx3-ubyte", data.TrainingCount);
		ProcessMNISTDataMT(10, data.ValidationInputs, data.ValidationTargets, "../../TestAndTraningSets/trainLabel.idx1-ubyte", "../../TestAndTraningSets/trainIm.idx3-ubyte", data.ValidationCount, 50000);
		ProcessMNISTDataMT(10, data.TestInputs, data.TestTargets, "../../TestAndTraningSets/testLabel.idx1-ubyte", "../../TestAndTraningSets/testIm.idx3-ubyte", data.TestCount);

		}

#endif

	
	SetupNetwork(data.TrainingInputs, data.TraningTargets, data.TrainingCount);


	pr("Before:");;
	TestNetwork(data.ValidationInputs, data.ValidationTargets, data.ValidationCount);

	TrainingLoop();

	pr("After");
	TestNetwork(data.ValidationInputs, data.ValidationTargets, data.ValidationCount);

	std::cin.get();
}



