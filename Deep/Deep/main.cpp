//#include "pch.h"
//#include "DataProcessing.h"



#define traningCount 50000

#define clean true



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

		ProcessMNISTDataMT(10, data.TrainingInputs, data.TraningTargets, "trainLabel.idx1-ubyte", "trainIm.idx3-ubyte", data.TrainingCount);
		ProcessMNISTDataMT(10, data.ValidationInputs, data.ValidationTargets, "trainLabel.idx1-ubyte", "trainIm.idx3-ubyte", data.ValidationCount, 50000);
		ProcessMNISTDataMT(10, data.TestInputs, data.TestTargets, "testLabel.idx1-ubyte", "testIm.idx3-ubyte", data.TestCount);

		}

#endif


	//std::cin.get();
}



