#include "pch.h"
#include "Layer.h"
#include "NetworkPrototype.h"

namespace TNNT
{
	void FullyConnectedLayer::FeedForward(FullyConnectedLayer* n)
	{

		//REMINDER: Weights, Biases and Z - arrays corresponding to layer n, are all saved on the n-1 spot in the m_WorkloadLayout array, but not in the m_LayerLayout array.

		LayerHeader* currentLayer = n->m_LayerLayoutPointer;
		LayerHeader* prevLayer = currentLayer->Prev();



		unsigned layerIndex = 0;
		while (layerIndex < currentLayer->NodesCount)
		{
			//float weightedSum = 0;



			//unsigned prevIndex = 0;
			//while (prevIndex < prevLayer->NodesCount)
			//{


			//	float prevA = prevLayer->A[prevIndex];
			//	float weight = currentLayer->Weights[prevLayer->NodesCount * layerIndex + prevIndex];
			//	weightedSum += weight * prevA;

			//	prevIndex++;
			//}


			//currentLayer->Z[layerIndex] = weightedSum + currentLayer->Biases[layerIndex];


			currentLayer->Z[layerIndex] = Math::Dot(&currentLayer->Weights[prevLayer->NodesCount * layerIndex], prevLayer->A, prevLayer->NodesCount) + currentLayer->Biases[layerIndex];

			currentLayer->A[layerIndex] = currentLayer->NeuronFunction(currentLayer->Z[layerIndex]);


			layerIndex++;
		}



	}
	void FullyConnectedLayer::Setup(char* storage)
	{



		Z = (float*)storage;

		dZ = Z + NodesCount;
		A = dZ + NodesCount;
		
		Weights = dZ + NodesCount;
		Biases = Weights + WeightsCount;
		
		WeightsTranspose = Biases + BiasesCount;
		

		TempWeights = WeightsTranspose + WeightsCount;
		TempBiases = TempWeights + WeightsCount;

		dWeights = TempBiases + BiasesCount;
		dBiases = dWeights + WeightsCount;
		

	}
	void FullyConnectedLayer::ResetTranspose()
	{

		unsigned n = H.NodesCount;
		//unsigned m = this->m;

			unsigned i = 0;
			while (i < n)
			{
				unsigned j = 0;
				while (j < Tm)
				{

					WeightsTranspose[ Tm* i + j] = Weights[n * j + i];
					j++;
				}
				i++;
			}

	}

	void FeedForward_FullyConnected()
	{

		//REMINDER: Weights, Biases and Z - arrays corresponding to layer n, are all saved on the n-1 spot in the m_WorkloadLayout array, but not in the m_LayerLayout array.

		LayerHeader* currentLayer = n->m_LayerLayoutPointer;
		LayerHeader* prevLayer = currentLayer->Prev();



		unsigned layerIndex = 0;
		while (layerIndex < currentLayer->NodesCount)
		{
			//float weightedSum = 0;



			//unsigned prevIndex = 0;
			//while (prevIndex < prevLayer->NodesCount)
			//{


			//	float prevA = prevLayer->A[prevIndex];
			//	float weight = currentLayer->Weights[prevLayer->NodesCount * layerIndex + prevIndex];
			//	weightedSum += weight * prevA;

			//	prevIndex++;
			//}


			//currentLayer->Z[layerIndex] = weightedSum + currentLayer->Biases[layerIndex];


			currentLayer->Z[layerIndex] = Math::Dot(&currentLayer->Weights[prevLayer->NodesCount * layerIndex], prevLayer->A, prevLayer->NodesCount) + currentLayer->Biases[layerIndex];

			currentLayer->A[layerIndex] = currentLayer->NeuronFunction(currentLayer->Z[layerIndex]);


			layerIndex++;
		}



	}
	void FeedForward_BackPropegation_FullyConnected()
	{


		LayerHeader* currentLayer = n->m_LayerLayoutPointer;
		LayerHeader* prevLayer = currentLayer->Prev();



		unsigned layerIndex = 0;
		while (layerIndex < currentLayer->NodesCount)
		{


			currentLayer->Z[layerIndex] = Math::Dot(&currentLayer->Weights[prevLayer->NodesCount * layerIndex], prevLayer->A, prevLayer->NodesCount) + currentLayer->Biases[layerIndex];

			currentLayer->A[layerIndex] = currentLayer->NeuronFunction(currentLayer->Z[layerIndex]);


			layerIndex++;
		}



	}
}