#include "pch.h"
#include "non_pch_includes.h"
#include "FullyConnectedLayer.h"




namespace TNNT {




float* FullyConnectedSetup(FullyConnectedHyperParams& n, float* storage, unsigned inputCount, unsigned outputCount, unsigned weightsCount, unsigned biasesCount)
{

	n.inputNodesCount = inputCount;
	n.outputNodesCount = outputCount;
	n.WeightsCount = weightsCount;
	n.BiasesCount = biasesCount;
	



	n.inputZ = storage;
	n.inputA = n.inputZ + n.inputNodesCount;

	n.inputDZ = n.inputA + n.inputNodesCount;
	n.inputDA = n.inputDZ + n.inputNodesCount;


	n.Weights = n.inputDA + n.inputNodesCount;
	n.Biases = n.Weights + n.WeightsCount;
	n.WeightsTranspose = n.Biases + n.BiasesCount;

	n.tempWeights = n.WeightsTranspose + n.WeightsCount;
	n.tempBiases = n.tempWeights + n.WeightsCount;

	n.DWeights = n.tempBiases + n.BiasesCount;
	n.DBiases = n.DWeights + n.WeightsCount;


	n.outputZ = n.DBiases + n.BiasesCount;
	n.outputA = n.outputZ + n.outputNodesCount;

	n.outputDZ = n.outputA + n.outputNodesCount;
	n.outputDA = n.outputDZ + n.outputNodesCount;

	float* endpoint = n.outputZ;



	//WEIGHTS AND BIASES SETUP START
	if (true)
	{
		//For randomly initializing the weights and biases
		std::default_random_engine generator;
		std::normal_distribution<float> distribution(0.0f, 1 / sqrt(n.inputNodesCount));


		unsigned index = 0;
		while (index < n.WeightsCount)
		{


			float temp = distribution(generator);
			n.Weights[index] = temp;

			index++;
		}




		index = 0;
		while (index < n.BiasesCount)
		{

			float temp = distribution(generator);
			n.Biases[index] = temp;

			index++;
		}



	}

	else
	{
		//Sets all weights and biases to zero

		unsigned index = 0;
		while (index < n.WeightsCount)
		{



			n.Weights[index] = 0;

			index++;
		}




		index = 0;
		while (index < n.BiasesCount)
		{


			n.Biases[index] = 0;

			index++;
		}
	}

	FullyConnectedSetTempToWeights(n);
	FullyConnectedSetTempToBiases(n);
	FullyConnectedResetTranspose(n);
	//WEIGHTS AND BIASES SETUP STOP



	//ENSURING THAT CERTAIN INTEGER AND FLOAT ARRAYS HAVE ACCEPTABLE INITIAL VALUES STOP

	return endpoint;
}

void FullyConnectedFeedForward(FullyConnectedHyperParams& n)
	{


		unsigned layerIndex = 0;
		while (layerIndex < n.outputNodesCount)
		{


			n.outputZ[layerIndex] = Math::Dot(&n.Weights[n.inputNodesCount* layerIndex], n.inputA, n.inputNodesCount) + n.Biases[layerIndex];

			n.outputA[layerIndex] = OutputNeuronFunction(n.outputZ[layerIndex]);

			layerIndex++;
		}



	}




	void FullyConnectedBackpropegateZ(FullyConnectedHyperParams& n)
	{






		unsigned layerIndex = 0;
		while (layerIndex < n.inputNodesCount)
		{




			float dAdZ = InputNeuronFunctionDerivative(n.inputZ[layerIndex]);

			n.inputDZ[layerIndex] = Math::Dot(&n.WeightsTranspose[n.outputNodesCount* layerIndex], n.outputDZ, n.outputNodesCount) * dAdZ;


			layerIndex++;
		}
		 
	}

	void FullyConnectedBackpropegateBW(FullyConnectedHyperParams& n)
	{



		unsigned layerIndex = 0;
		while (layerIndex < n.outputNodesCount)
		{


			const float dz = n.outputDZ[layerIndex];

			n.DBiases[layerIndex] = dz;



			memcpy(&n.DWeights[n.inputNodesCount * layerIndex], n.inputA, n.inputNodesCount * sizeof(float));
			Math::ScalarMult(&n.DWeights[n.inputNodesCount * layerIndex], dz, n.inputNodesCount);

			layerIndex++;
		}

	}
	void FullyConnectedResetTranspose(FullyConnectedHyperParams& n)
	{
		unsigned inputIndex = 0;
		while (inputIndex < n.inputNodesCount)
		{
			unsigned outputIndex = 0;
			while (outputIndex < n.outputNodesCount)
			{

				n.WeightsTranspose[n.outputNodesCount * inputIndex + outputIndex] = n.Weights[n.inputNodesCount * outputIndex + inputIndex];
				outputIndex++;
			}
			inputIndex++;
		}
	}
	void FullyConnectedSetWeightsToTemp(FullyConnectedHyperParams& n)
	{
		memcpy(n.Weights, n.tempWeights, sizeof(float) * n.WeightsCount);

		FullyConnectedResetTranspose(n);
	}
	void FullyConnectedSetTempToWeights(FullyConnectedHyperParams& n)
	{
		memcpy(n.tempWeights, n.Weights, sizeof(float) * n.WeightsCount);
	}
	void FullyConnectedSetBiasesToTemp(FullyConnectedHyperParams& n)
	{
		memcpy(n.Biases, n.tempBiases, sizeof(float) * n.BiasesCount);
	}
	void FullyConnectedSetTempToBiases(FullyConnectedHyperParams& n)
	{
		memcpy(n.tempBiases, n.Biases, sizeof(float) * n.BiasesCount);
	}	
}