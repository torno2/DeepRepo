#include "pch.h"
#include "FullyConnectedLayer.h"
#include "NetworkPrototype.h"



#define InputNeuronFunction(x) Math::Sigmoid(x)
#define InputNeuronFunctionDerivative(x) Math::SigmoidDerivative(x)

#define OutputNeuronFunction(x) Math::Sigmoid(x)
#define OutputNeuronFunctionDerivative(x) Math::SigmoidDerivative(x)

namespace TNNT {

	struct FullyConnectedHyperParams
	{

		float* inputZ;
		float* inputA;

		float* outputZ;
		float* outputA;


		float* Weights;
		float* Biases;
		float* WeightsTranspose;

		float* inputDZ;
		float* inputDA;

		float* outputDA;
		float* outputDZ;

		float* DWeights;
		float* DBiases;


		unsigned inputNodesCount;
		unsigned outputNodesCount;
		

		unsigned WeightsCount;
		unsigned BiasesCount;
	};


\


	void FullyConnectedFeedForward(FullyConnectedHyperParams n)
	{


		unsigned layerIndex = 0;
		while (layerIndex < n.outputNodesCount)
		{


			n.outputZ[layerIndex] = Math::Dot(&n.Weights[n.inputNodesCount* layerIndex], n.inputA, n.inputNodesCount) + n.Biases[layerIndex];

			n.outputA[layerIndex] = OutputNeuronFunction(n.outputZ[layerIndex]);

			layerIndex++;
		}



	}




	void FullyConnectedBackpropegateZ(FullyConnectedHyperParams n)
	{






		unsigned layerIndex = 0;
		while (layerIndex < n.inputNodesCount)
		{




			float dAdZ = InputNeuronFunctionDerivative(n.inputZ[layerIndex]);

			n.inputDZ[layerIndex] = Math::Dot(&n.WeightsTranspose[n.outputNodesCount* layerIndex], n.outputDZ, n.outputNodesCount) * dAdZ;


			layerIndex++;
		}
		 
	}

	void FullyConnectedBackpropegateBW(FullyConnectedHyperParams n)
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
}