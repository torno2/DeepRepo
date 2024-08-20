#include "pch.h"
#include "LayerFunctionsMT.h"

namespace TNNT
{

	namespace LayerFunctionsMT
	{
		

		void FullyConnectedFeedForward(NetworkPrototypeMT* n, unsigned thread)
		{

			//REMINDER: Weights, Biases and Z - arrays corresponding to layer n, are all saved on the n-1 spot in the m_WorkloadLayout array, but not in the m_LayerLayout array.

			LayerLayout prevLayer = n->m_LayerLayout[n->m_LayerLayoutPosition - 1];
			LayerLayout currentLayer = n->m_LayerLayout[n->m_LayerLayoutPosition];


			unsigned start = n->m_WorkloadLayout.Nodes[2 * thread + (2 * n->m_SlaveThreadCount * (n->m_LayerLayoutPosition))];
			unsigned stop = n->m_WorkloadLayout.Nodes[1 + 2 * thread + (2 * n->m_SlaveThreadCount * (n->m_LayerLayoutPosition))];


			unsigned layerIndex = start;
			while (layerIndex < stop)
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

				currentLayer.A[layerIndex] = n->m_Functions.NeuronFunctions[n->m_LayerLayoutPosition - 1].f(currentLayer.Z[layerIndex]);


				layerIndex++;
			}


		}

		void FullyConnectedBackpropegateZ(NetworkPrototypeMT* n, unsigned thread)
		{

			//REMINDER: Weights, Biases and Z - arrays corresponding to layer n, are all saved on the n-1 spot in the m_WorkloadLayout array, but not in the m_LayerLayout array.


			LayerLayout currentLayer = n->m_LayerLayout[n->m_LayerLayoutPosition];
			LayerLayout latterLayer = n->m_LayerLayout[n->m_LayerLayoutPosition+1];


			unsigned start = n->m_WorkloadLayout.Nodes[2 * thread + (2 * n->m_SlaveThreadCount * (n->m_LayerLayoutPosition))];
			unsigned stop = n->m_WorkloadLayout.Nodes[1 + 2 * thread + (2 * n->m_SlaveThreadCount * (n->m_LayerLayoutPosition))];



			unsigned layerIndex = start;
			while (layerIndex < stop)
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



				float dz = n->m_Functions.NeuronFunctionsDerivatives[n->m_LayerLayoutPosition-1].f(currentLayer.Z[layerIndex]);
				currentLayer.dZ[layerIndex] = errorSum * dz;


				layerIndex++;
			}

		}

		void FullyConnectedBackpropegateBW(NetworkPrototypeMT* n, unsigned thread)
		{

			//REMINDER: Weights, Biases and Z - arrays corresponding to layer n, are all saved on the n-1 spot in the m_WorkloadLayout array, but not in the m_LayerLayout array.


			LayerLayout prevLayer = n->m_LayerLayout[n->m_LayerLayoutPosition - 1];
			LayerLayout currentLayer = n->m_LayerLayout[n->m_LayerLayoutPosition];




			unsigned startNodes = n->m_WorkloadLayout.Nodes[(2 * thread) + (2 * n->m_SlaveThreadCount * n->m_LayerLayoutPosition)];
			unsigned stopNodes = n->m_WorkloadLayout.Nodes[(2 * thread + 1) + (2 * n->m_SlaveThreadCount * n->m_LayerLayoutPosition)];


			unsigned layerIndex = startNodes;
			while (layerIndex < stopNodes)
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


	}



	namespace CostFunctionsMT
	{
		void CrossEntropy(NetworkPrototypeMT* n, unsigned thread)
		{

			//REMINDER: Weights, Biases and Z - arrays corresponding to layer n, are all saved on the n-1 spot in the m_WorkloadLayout array, but not in the m_LayerLayout array.

			unsigned start = n->m_WorkloadLayout.Output[2*thread ];
			unsigned stop = n->m_WorkloadLayout.Output[2*thread+1];

			unsigned layerIndex = start;
			while (layerIndex < stop)
			{
				
				float a = n->m_OutputBuffer[layerIndex];
				float y = n->m_TargetBuffer[layerIndex];


				float cost = Math::CrossEntropy(a, y);


				n->m_CostBuffer[thread] += cost; 
				


				layerIndex++;
			}
		}

		void CrossEntropyDerivative(NetworkPrototypeMT* n, unsigned thread)
		{

			LayerLayout currentLayer = n->m_LayerLayout[n->m_LayerLayoutPosition];

			//REMINDER: Weights, Biases and Z - arrays corresponding to layer n, are all saved on the n-1 spot in the m_WorkloadLayout array, but not in the m_LayerLayout array.

			unsigned start = n->m_WorkloadLayout.Output[2 * thread];
			unsigned stop = n->m_WorkloadLayout.Output[2 * thread + 1];
		

			unsigned layerIndex = start;
			while (layerIndex < stop)
			{
				
				float z = currentLayer.Z[layerIndex];
				float a = n->m_OutputBuffer[layerIndex];
				float y = n->m_TargetBuffer[layerIndex];

				float dz = Math::CrossEntropyCostDerivative(z, a, y);
				
				currentLayer.dZ[layerIndex] = dz;
				

				layerIndex++;
			}

		}


	}


	namespace TrainingFunctionsMT
	{

		void L2Regularization(NetworkPrototypeMT* n, unsigned thread)
		{

			//REMINDER: Weights, BViases and Z - arrays corresponding to layer n, are all saved on the n-1 spot in the m_WorkloadLayout array, but not in the m_LayerLayout array.

			//Always remember that the 0-th layer doesnt have any weights and biases.

			unsigned layerPos = n->m_LayerLayoutPosition;
			LayerLayout currentLayer = n->m_LayerLayout[layerPos];

			unsigned start = n->m_WorkloadLayout.Weights[   2 * thread + (layerPos - 1) * (2 * n->m_SlaveThreadCount)];
			unsigned stop = n->m_WorkloadLayout.Weights[2 * thread + 1 + (layerPos - 1) * (2 * n->m_SlaveThreadCount)];
			
			

			unsigned index = start;
			while (index < stop)
			{




				currentLayer.TempWeights[index] *= (1 - (currentLayer.LearningRate * currentLayer.RegularizationConstant / ((float)n->m_Data->TrainingCount)));
				
				auto temp = currentLayer.TempWeights[index];

				index++;


			}

		}


		void GradientDecent(NetworkPrototypeMT* n, unsigned thread)
		{
			//REMINDER: Weights, Biases and Z - arrays corresponding to layer n, are all saved on the n-1 spot in the m_WorkloadLayout array, but not in the m_LayerLayout array.

			//Always remember that the 0-th layer doesnt have any weights and biases, which means that the amount of layers you need to count is: m_layerLayoutCount-1

			unsigned layerPos = n->m_LayerLayoutPosition;
			LayerLayout currentLayer = n->m_LayerLayout[layerPos];

			unsigned wStart = n->m_WorkloadLayout.Weights[2 * thread + (layerPos - 1) * (2 * n->m_SlaveThreadCount)];
			unsigned wStop = n->m_WorkloadLayout.Weights[2 * thread + 1 + (layerPos - 1) * (2 * n->m_SlaveThreadCount)];

			unsigned bStart = n->m_WorkloadLayout.Biases[2 * thread + (layerPos - 1) * (2 * n->m_SlaveThreadCount)];
			unsigned bStop = n->m_WorkloadLayout.Biases[2 * thread + 1 + (layerPos - 1) * (2 * n->m_SlaveThreadCount)];

			
			unsigned wIndex = wStart;
			while (wIndex < wStop)
			{
				


				float temp = (currentLayer.LearningRate / ((float)n->m_HyperParameters.BatchCount)) * currentLayer.dWeights[wIndex];
				currentLayer.TempWeights[wIndex] -= temp;


				wIndex++;
			}


			unsigned bIndex = bStart;
			while (bIndex < bStop)
			{

				float tempB = (currentLayer.LearningRate / ((float)n->m_HyperParameters.BatchCount)) * currentLayer.dBiases[bIndex];
				currentLayer.TempBiases[bIndex] -= tempB;

				bIndex++;
				
			}

		}


	}


}