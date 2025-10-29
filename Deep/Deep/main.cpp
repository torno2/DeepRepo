#include "pch.h"
#include "non_pch_includes.h"

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


	constexpr unsigned labelSize = 10;
	constexpr unsigned inputSize = 28 * 28;

	unsigned TrainingCount = traningCount;
	float* TrainingInputs = new float[TrainingCount * inputSize];
	float* TraningTargets = new float[TrainingCount * labelSize];

	unsigned ValidationCount = 10000;
	float* ValidationInputs = new float[ValidationCount * inputSize];
	float* ValidationTargets = new float[ValidationCount * labelSize];

	unsigned TestCount = 10000;
	float* TestInputs = new float[TestCount * inputSize];
	float* TestTargets = new float[TestCount * labelSize];

	//Data Formating start

	ProcessMNISTDataMT(10, TrainingInputs, TraningTargets, "../../TestAndTraningSets/trainLabel.idx1-ubyte", "../../TestAndTraningSets/trainIm.idx3-ubyte", TrainingCount);
	ProcessMNISTDataMT(10, ValidationInputs, ValidationTargets, "../../TestAndTraningSets/trainLabel.idx1-ubyte", "../../TestAndTraningSets/trainIm.idx3-ubyte", ValidationCount, 50000);
	ProcessMNISTDataMT(10, TestInputs, TestTargets, "../../TestAndTraningSets/testLabel.idx1-ubyte", "../../TestAndTraningSets/testIm.idx3-ubyte", TestCount);
	pr("Datafomating time: " << t.Stop());

		
	t.Start();
	SetupNetwork(TrainingInputs, TraningTargets, TrainingCount);
	pr("Setup time: " << t.Stop());
	

	t.Start();
	TrainingLoop();
	pr("Training time: " << t.Stop());

	t.Start();
	TestNetwork(ValidationInputs, ValidationTargets, ValidationCount);
	pr("Test time: " << t.Stop());

	std::cin.get();

	
#endif

	

}



