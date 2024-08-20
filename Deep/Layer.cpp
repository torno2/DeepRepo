#include "Layer.h"
#include "NetworkPrototype2.h"
#include "Math.h"
namespace TNNT
{
	float FCLayer::NeuronFunction(float z)
	{
		return Math::Sigmoid(z);
	}
	float FCLayer::NeuronFunctionDerivative(float z)
	{
		return Math::SigmoidDerivative(z);
	}
	void FCLayer::FeedForward(NetworkPrototype2* n)
	{
		//REMINDER: Weights, Biases and Z - arrays corresponding to layer n, are all saved on the n-1 spot in the m_WorkloadLayout array, but not in the m_LayerLayout array.

		FCLayer prevLayer = n->m_LayerLayout[n->m_LayerLayoutPosition - 1];
		FCLayer currentLayer = n->m_LayerLayout[n->m_LayerLayoutPosition];


		float* weights;

		unsigned layerIndex = 0;
		while (layerIndex < currentLayer.NodesCount)
		{
			float weightedSum = 0;



			unsigned prevIndex = 0;
			while (prevIndex < prevLayer.NodesCount)
			{


				float prevA = prevLayer.A[prevIndex];
				float weight = currentLayer.Weights[prevLayer.NodesCount * layerIndex + prevIndex];
				weightedSum += weight * prevA;

				prevIndex++;
			}


			currentLayer.Z[layerIndex] = weightedSum + currentLayer.Biases[layerIndex];

			currentLayer.A[layerIndex] = NeuronFunction(currentLayer.Z[layerIndex]);


			layerIndex++;
		}
	}
	void FCLayer::BackPropegateZ(NetworkPrototype2* n)
	{

		FCLayer currentLayer = n->m_LayerLayout[n->m_LayerLayoutPosition];
		FCLayer latterLayer = n->m_LayerLayout[n->m_LayerLayoutPosition + 1];




		unsigned layerIndex = 0;
		while (layerIndex < currentLayer.NodesCount)
		{

			float errorSum = 0;
			unsigned latterLayerIndex = 0;
			while (latterLayerIndex < latterLayer.NodesCount)
			{
				float latterWeight = latterLayer.Weights[currentLayer.NodesCount * latterLayerIndex + layerIndex];
				float latterDZ = latterLayer.dZ[latterLayerIndex];

				errorSum += latterWeight * latterDZ;



				latterLayerIndex++;
			}



			float dAdZ = NeuronFunctionDerivative(currentLayer.Z[layerIndex]);
			currentLayer.dZ[layerIndex] = errorSum * dAdZ;


			layerIndex++;
		}
	}
	void FCLayer::BackPropegateBW(NetworkPrototype2* n)
	{

		//REMINDER: Weights, Biases and Z - arrays corresponding to layer n, are all saved on the n-1 spot in the m_WorkloadLayout array, but not in the m_LayerLayout array.


		FCLayer prevLayer = n->m_LayerLayout[n->m_LayerLayoutPosition - 1];
		FCLayer currentLayer = n->m_LayerLayout[n->m_LayerLayoutPosition];


		unsigned layerIndex = 0;
		while (layerIndex < currentLayer.NodesCount)
		{


			const float dz = currentLayer.dZ[layerIndex];

			currentLayer.dBiases[layerIndex] = dz;



			unsigned prevLayerIndex = 0;
			while (prevLayerIndex < prevLayer.NodesCount)
			{



				float a = prevLayer.A[prevLayerIndex];
				float dw = a * dz;


				currentLayer.dWeights[prevLayer.NodesCount * layerIndex + prevLayerIndex] = dw;




				prevLayerIndex++;
			}

			layerIndex++;
		}
	}
	void FCLayer::Regularize(NetworkPrototype2* n)
	{
		//REMINDER: Weights, BViases and Z - arrays corresponding to layer n, are all saved on the n-1 spot in the m_WorkloadLayout array, but not in the m_LayerLayout array.

		//Always remember that the 0-th layer doesnt have any weights and biases.

		unsigned layerPos = n->m_LayerLayoutPosition;
		FCLayer currentLayer = n->m_LayerLayout[layerPos];

		unsigned index = 0;
		while (index < currentLayer.NodesCount)
		{




			currentLayer.TempWeights[index] *= (1 - (currentLayer.LearningRate * currentLayer.RegularizationConstant / ((float)n->m_Data->TrainingCount)));

			auto temp = currentLayer.TempWeights[index];

			index++;


		}
	}
	void FCLayer::Train(NetworkPrototype2* n)
	{
		//REMINDER: Weights, Biases and Z - arrays corresponding to layer n, are all saved on the n-1 spot in the m_WorkloadLayout array, but not in the m_LayerLayout array.

		//Always remember that the 0-th layer doesnt have any weights and biases, which means that the amount of layers you need to count is: m_layerLayoutCount-1

		unsigned layerPos = n->m_LayerLayoutPosition;
		FCLayer currentLayer = n->m_LayerLayout[layerPos];




		unsigned index = 0;
		while (index < currentLayer.WeightsCount)
		{





			float temp = (currentLayer.LearningRate / ((float)n->m_HyperParameters.BatchCount)) * currentLayer.dWeights[index];
			currentLayer.TempWeights[index] -= temp;


			index++;
		}

		index = 0;
		while (index < currentLayer.BiasesCount)
		{

			float tempB = (currentLayer.LearningRate / ((float)n->m_HyperParameters.BatchCount)) * currentLayer.dBiases[index];
			currentLayer.TempBiases[index] -= tempB;

			index++;

		}
	}
}