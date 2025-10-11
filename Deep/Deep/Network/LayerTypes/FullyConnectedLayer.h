#pragma once


#ifdef FullyConnectedLayerDef


namespace TNNT {

	struct FullyConnectedHyperParams
	{

		float* inputZ;
		float* inputA;

		float* inputDZ;
		float* inputDA;



		float* Weights;
		float* Biases;
		float* WeightsTranspose;

		float* tempWeights;
		float* tempBiases;

		float* DWeights;
		float* DBiases;


		float* outputZ;
		float* outputA;

		float* outputDA;
		float* outputDZ;




		unsigned inputNodesCount;
		unsigned outputNodesCount;

		unsigned WeightsCount;
		unsigned BiasesCount;
	};


	float* FullyConnectedSetup(FullyConnectedHyperParams n, float* storage);

	void FullyConnectedFeedForward(FullyConnectedHyperParams n);

	void FullyConnectedBackpropegateZ(FullyConnectedHyperParams n);
	void FullyConnectedBackpropegateBW(FullyConnectedHyperParams n);


	void FullyConnectedResetTranspose(FullyConnectedHyperParams n);

	void FullyConnectedSetWeightsToTemp(FullyConnectedHyperParams n);
	void FullyConnectedSetTempToWeights(FullyConnectedHyperParams n);
	void FullyConnectedSetBiasesToTemp(FullyConnectedHyperParams n);
	void FullyConnectedSetTempToBiases(FullyConnectedHyperParams n);





}


#endif