#include "pch.h"
#include "LayerFunctions.h"

namespace TNNT
{
	namespace LayerFunctions
	{
		
		void FullyConnectedFeedForward(NetworkPrototype * n)
		{

			//REMINDER: Weights, Biases and Z - arrays corresponding to layer n, are all saved on the n-1 spot in the m_WorkloadLayout array, but not in the m_LayerLayout array.

			LayerLayout prevLayer = n->m_LayerLayout[n->m_LayerLayoutPosition - 1];
			LayerLayout currentLayer = n->m_LayerLayout[n->m_LayerLayoutPosition];


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

				currentLayer.A[layerIndex] = n->m_Functions.NeuronFunctions[n->m_LayerLayoutPosition - 1].f(currentLayer.Z[layerIndex]);


				layerIndex++;
			}



		}

		void FullyConnectedBackpropegateZ(NetworkPrototype * n)
		{


			//REMINDER: Weights, Biases and Z - arrays corresponding to layer n, are all saved on the n-1 spot in the m_WorkloadLayout array, but not in the m_LayerLayout array.


			LayerLayout currentLayer = n->m_LayerLayout[n->m_LayerLayoutPosition];
			LayerLayout latterLayer = n->m_LayerLayout[n->m_LayerLayoutPosition + 1];




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



				float dAdZ = n->m_Functions.NeuronFunctionsDerivatives[n->m_LayerLayoutPosition - 1].f(currentLayer.Z[layerIndex]);
				currentLayer.dZ[layerIndex] = errorSum * dAdZ;


				layerIndex++;
			}

		}

		void FullyConnectedBackpropegateBW(NetworkPrototype * n)
		{


			//REMINDER: Weights, Biases and Z - arrays corresponding to layer n, are all saved on the n-1 spot in the m_WorkloadLayout array, but not in the m_LayerLayout array.


			LayerLayout prevLayer = n->m_LayerLayout[n->m_LayerLayoutPosition - 1];
			LayerLayout currentLayer = n->m_LayerLayout[n->m_LayerLayoutPosition];


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

		


		void ConvolutionLayerFeedForward(NetworkPrototype* n)
		{
			
			

			LayerLayout prevLayer = n->m_LayerLayout[n->m_LayerLayoutPosition-1];
			LayerLayout currentLayer = n->m_LayerLayout[n->m_LayerLayoutPosition];

			//Used with the "TensorOverlay" class to navigate the node array of prevLayer as an N-dimentional box.
			unsigned* posBuffer = new unsigned[currentLayer.KerDimCount];
			unsigned* stridePosBuffer = new unsigned[currentLayer.KerDimCount];
			



			//Node count of individual feature maps
			unsigned featureMapNodeCount = currentLayer.NodesCount / currentLayer.SubLayerCount;
			unsigned featureMapWeightsCount = currentLayer.WeightsCount / currentLayer.SubLayerCount;
			unsigned featureMapWeightLinesCount = featureMapWeightsCount / currentLayer.KerDim[0];
			unsigned featureMapBiasesCount = currentLayer.BiasesCount / currentLayer.SubLayerCount;

			unsigned featureMap = 0;
			while (featureMap < currentLayer.SubLayerCount)
			{

				memset(posBuffer, 0, currentLayer.KerDimCount*sizeof(unsigned));
				memset(stridePosBuffer, 0, currentLayer.KerDimCount * sizeof(unsigned));


				float* A = currentLayer.A + featureMap * featureMapNodeCount;
				float* Z = currentLayer.Z + featureMap * featureMapNodeCount;
				float* weights = currentLayer.Weights + featureMap * featureMapWeightsCount;
				float* biases = currentLayer.Biases + featureMap * featureMapBiasesCount;

				TensorOverlay prevA(prevLayer.A, prevLayer.NodesCount, prevLayer.LayerDim, prevLayer.LayerDimCount);
				
				unsigned featureMapIndex = 0;
				while (featureMapIndex < featureMapNodeCount)
				{
					float weightedSum = 0;

					
					unsigned kerLine = 0;
					while (kerLine < featureMapWeightLinesCount)
					{
						
						float* aLine = prevA.At(posBuffer);
						unsigned kerLineIndex = 0;
						while (kerLineIndex < currentLayer.KerDim[0])
						{
							float weight = weights[kerLineIndex + kerLine * currentLayer.KerDim[0]];
							float a = aLine[kerLineIndex];

							weightedSum += weight * a;

							kerLineIndex++;
						}

						kerLine++;
						

						unsigned kerDimMult = 1;
						unsigned kerDimIndex = 1;
						while (kerDimIndex < currentLayer.KerDimCount)
						{
							if (kerDimIndex == 1)
							{
								posBuffer[1]++;
							}
							
							if (  ((posBuffer[kerDimIndex] - stridePosBuffer[kerDimIndex])) > currentLayer.KerDim[kerDimIndex] )
							{
								posBuffer[kerDimIndex]++;
								posBuffer[kerDimIndex - 1] = 0 + stridePosBuffer[kerDimIndex - 1];
								kerDimMult *= currentLayer.KerDim[kerDimIndex];

							}

							else
							{
								break;
							}
							kerDimIndex++;
						}
						

						
					}
					

					stridePosBuffer[0] += currentLayer.Stride[0];
					unsigned kerDimIndex = 0;
					while(kerDimIndex < currentLayer.KerDimCount-1)
					{

						if ( (stridePosBuffer[kerDimIndex] + (currentLayer.KerDim[kerDimIndex]-1)) >= (prevLayer.LayerDim[kerDimIndex ]))
						{

							stridePosBuffer[kerDimIndex] = 0;
							stridePosBuffer[kerDimIndex + 1] += currentLayer.Stride[kerDimIndex + 1];


						}

						posBuffer[kerDimIndex] = 0 + stridePosBuffer[kerDimIndex];
						posBuffer[kerDimIndex + 1] = 0 + stridePosBuffer[kerDimIndex + 1];

						kerDimIndex++;
					}
					
					


					


					weightedSum += *biases;

					float z = weightedSum;
					Z[featureMapIndex] = z;
					float a = n->m_Functions.NeuronFunctions[n->m_LayerLayoutPosition - 1].f(z);
					A[featureMapIndex] = a;

					
					featureMapIndex++;
				}

				

				featureMap++;
			}


			delete[] posBuffer;
			delete[] stridePosBuffer;


		}

		void ConvolutionLayerBackpropegateZ(NetworkPrototype* n)
		{
			LayerLayout currentLayer = n->m_LayerLayout[n->m_LayerLayoutPosition];
			LayerLayout latterLayer = n->m_LayerLayout[n->m_LayerLayoutPosition+1];

			//Used with the "TensorOverlay" class to navigate the node array of currentLayer as an N-dimentional box.
			unsigned* posBuffer = new unsigned[latterLayer.KerDimCount];
			unsigned* stridePosBuffer = new unsigned[latterLayer.KerDimCount];

			//Set the delta weights array
			memset(currentLayer.dZ, 0.0, currentLayer.ZCount * sizeof(float));


			//Node count of individual feature maps
			unsigned featureMapNodeCount = latterLayer.NodesCount / latterLayer.SubLayerCount;
			unsigned featureMapWeightsCount = latterLayer.WeightsCount / latterLayer.SubLayerCount;
			unsigned featureMapWeightLinesCount = featureMapWeightsCount / latterLayer.KerDim[0];
			unsigned featureMapBiasesCount = latterLayer.BiasesCount / latterLayer.SubLayerCount;

			unsigned featureMap = 0;
			while (featureMap < latterLayer.SubLayerCount)
			{

				memset(posBuffer, 0, latterLayer.KerDimCount * sizeof(unsigned));
				memset(stridePosBuffer, 0, latterLayer.KerDimCount * sizeof(unsigned));

				
				
				float* latterDZ = latterLayer.dZ + featureMap * featureMapNodeCount;
				float* weights = latterLayer.Weights + featureMap * featureMapWeightsCount;
				

				TensorOverlay dZ(currentLayer.dZ, currentLayer.ZCount, currentLayer.LayerDim, currentLayer.LayerDimCount);
				TensorOverlay Z(currentLayer.Z, currentLayer.ZCount, currentLayer.LayerDim, currentLayer.LayerDimCount);

				unsigned featureMapIndex = 0;
				while (featureMapIndex < featureMapNodeCount)
				{

					float latterdz = latterDZ[featureMapIndex];
					


					unsigned kerLine = 0;
					while (kerLine < featureMapWeightLinesCount)
					{

						float* dZLine = dZ.At(posBuffer);
						float* ZLine = Z.At(posBuffer);
						unsigned kerLineIndex = 0;
						while (kerLineIndex < latterLayer.KerDim[0])
						{
							
							
							float weight = weights[kerLineIndex + kerLine * latterLayer.KerDim[0]];
							float dAdZ = n->m_Functions.NeuronFunctionsDerivatives[n->m_LayerLayoutPosition - 1].f(ZLine[kerLineIndex]);

							dZLine[kerLineIndex] += dAdZ *weight * latterdz;

							kerLineIndex++;
						}

						kerLine++;


						unsigned kerDimMult = 1;
						unsigned kerDimIndex = 1;
						while (kerDimIndex < latterLayer.KerDimCount)
						{
							if (kerDimIndex == 1)
							{
								posBuffer[1]++;
							}

							if (((posBuffer[kerDimIndex] - stridePosBuffer[kerDimIndex])) > latterLayer.KerDim[kerDimIndex])
							{
								posBuffer[kerDimIndex]++;
								posBuffer[kerDimIndex - 1] = 0 + stridePosBuffer[kerDimIndex - 1];
								kerDimMult *= latterLayer.KerDim[kerDimIndex];

							}

							else
							{
								break;
							}
							kerDimIndex++;
						}



					}


					stridePosBuffer[0] += latterLayer.Stride[0];
					unsigned kerDimIndex = 0;
					while (kerDimIndex < latterLayer.KerDimCount - 1)
					{

						if ((stridePosBuffer[kerDimIndex] + (latterLayer.KerDim[kerDimIndex] - 1)) >= (currentLayer.LayerDim[kerDimIndex]))
						{

							stridePosBuffer[kerDimIndex] = 0;
							stridePosBuffer[kerDimIndex + 1] += latterLayer.Stride[kerDimIndex + 1];



						}

						posBuffer[kerDimIndex] = 0 + stridePosBuffer[kerDimIndex];
						posBuffer[kerDimIndex + 1] = 0 + stridePosBuffer[kerDimIndex + 1];

						kerDimIndex++;
					}












					featureMapIndex++;
				}



				featureMap++;
			}


			delete[] posBuffer;
			delete[] stridePosBuffer;
		}

		void ConvolutionLayerBackpropegateBW(NetworkPrototype* n)
		{
			LayerLayout prevLayer = n->m_LayerLayout[n->m_LayerLayoutPosition - 1];
			LayerLayout currentLayer = n->m_LayerLayout[n->m_LayerLayoutPosition];

			//Used with the "TensorOverlay" class to navigate the node array of prevLayer as an N-dimentional box.
			unsigned* posBuffer = new unsigned[currentLayer.KerDimCount];
			unsigned* stridePosBuffer = new unsigned[currentLayer.KerDimCount];

			//Set the delta weights array
			memset(currentLayer.dWeights, 0.0, currentLayer.WeightsCount*sizeof(float));
			memset(currentLayer.dBiases, 0.0, currentLayer.BiasesCount*sizeof(float));


			//Node count of individual feature maps
			unsigned featureMapNodeCount = currentLayer.NodesCount / currentLayer.SubLayerCount;
			unsigned featureMapWeightsCount = currentLayer.WeightsCount / currentLayer.SubLayerCount;
			unsigned featureMapWeightLinesCount = featureMapWeightsCount / currentLayer.KerDim[0];
			unsigned featureMapBiasesCount = currentLayer.BiasesCount / currentLayer.SubLayerCount;

			unsigned featureMap = 0;
			while (featureMap < currentLayer.SubLayerCount)
			{
				
				memset(posBuffer, 0, currentLayer.KerDimCount*sizeof(unsigned));
				memset(stridePosBuffer, 0, currentLayer.KerDimCount*sizeof(unsigned));

				float* dZ = currentLayer.dZ + featureMap * featureMapNodeCount;
				float* dweights = currentLayer.dWeights + featureMap * featureMapWeightsCount;
				float* dbiases = currentLayer.dBiases + featureMap * featureMapBiasesCount;

				TensorOverlay prevA(prevLayer.A, prevLayer.NodesCount, prevLayer.LayerDim, prevLayer.LayerDimCount);

				unsigned featureMapIndex = 0;
				while (featureMapIndex < featureMapNodeCount)
				{

					float deltaZ = dZ[featureMapIndex];
					*dbiases += deltaZ;
					
					
					

					unsigned kerLine = 0;
					while (kerLine < featureMapWeightLinesCount)
					{

						float* aLine = prevA.At(posBuffer);
						unsigned kerLineIndex = 0;
						while (kerLineIndex < currentLayer.KerDim[0])
						{
							float a = aLine[kerLineIndex];

							dweights[kerLineIndex + kerLine * currentLayer.KerDim[0]] += a * deltaZ;
							


							kerLineIndex++;
						}

						kerLine++;


						unsigned kerDimMult = 1;
						unsigned kerDimIndex = 1;
						while (kerDimIndex < currentLayer.KerDimCount)
						{
							if (kerDimIndex == 1)
							{
								posBuffer[1]++;
							}

							if (((posBuffer[kerDimIndex] - stridePosBuffer[kerDimIndex])) > currentLayer.KerDim[kerDimIndex])
							{
								posBuffer[kerDimIndex]++;
								posBuffer[kerDimIndex - 1] = 0 + stridePosBuffer[kerDimIndex - 1];
								kerDimMult *= currentLayer.KerDim[kerDimIndex];

							}

							else
							{
								break;
							}
							kerDimIndex++;
						}



					}


					stridePosBuffer[0] += currentLayer.Stride[0];
					unsigned kerDimIndex = 0;
					while (kerDimIndex < currentLayer.KerDimCount - 1)
					{

						if ((stridePosBuffer[kerDimIndex] + (currentLayer.KerDim[kerDimIndex] - 1)) >= (prevLayer.LayerDim[kerDimIndex]))
						{

							stridePosBuffer[kerDimIndex] = 0;
							stridePosBuffer[kerDimIndex + 1] += currentLayer.Stride[kerDimIndex + 1];




						}

						posBuffer[kerDimIndex] = 0 + stridePosBuffer[kerDimIndex];
						posBuffer[kerDimIndex + 1] = 0 + stridePosBuffer[kerDimIndex + 1];

						kerDimIndex++;
					}







					




					featureMapIndex++;
				}



				featureMap++;
			}


			delete[] posBuffer;
			delete[] stridePosBuffer;

		}

	

		void PoolingLayerFeedForward(NetworkPrototype* n)
		{



			LayerLayout prevLayer = n->m_LayerLayout[n->m_LayerLayoutPosition - 1];
			LayerLayout currentLayer = n->m_LayerLayout[n->m_LayerLayoutPosition];

			//Used with the "TensorOverlay" class to navigate the node array of prevLayer as an N-dimentional box.
			unsigned* posBuffer = new unsigned[currentLayer.KerDimCount];
			unsigned* stridePosBuffer = new unsigned[currentLayer.KerDimCount];




			//Node count of individual feature maps
			unsigned featureMapNodeCount = currentLayer.NodesCount / currentLayer.SubLayerCount;
			unsigned featureMapWeightsCount = currentLayer.WeightsCount / currentLayer.SubLayerCount;
			unsigned featureMapWeightLinesCount = featureMapWeightsCount / currentLayer.KerDim[0];
			unsigned featureMapBiasesCount = currentLayer.BiasesCount / currentLayer.SubLayerCount;

			unsigned featureMap = 0;
			while (featureMap < currentLayer.SubLayerCount)
			{

				memset(posBuffer, 0, currentLayer.KerDimCount * sizeof(unsigned));
				memset(stridePosBuffer, 0, currentLayer.KerDimCount * sizeof(unsigned));


				float* A = currentLayer.A + featureMap * featureMapNodeCount;
				float* Z = currentLayer.Z + featureMap * featureMapNodeCount;

				TensorOverlay prevA(prevLayer.A, prevLayer.NodesCount, prevLayer.LayerDim, prevLayer.LayerDimCount);

				unsigned featureMapIndex = 0;
				while (featureMapIndex < featureMapNodeCount)
				{
					
					float max = 0;

					unsigned kerLine = 0;
					while (kerLine < featureMapWeightLinesCount)
					{
						

						float* aLine = prevA.At(posBuffer);
						unsigned kerLineIndex = 0;
						while (kerLineIndex < currentLayer.KerDim[0])
						{
							
							float a = aLine[kerLineIndex];

							if (a > max)
							{
								max = a;
							}

							kerLineIndex++;
						}

						kerLine++;


						unsigned kerDimMult = 1;
						unsigned kerDimIndex = 1;
						while (kerDimIndex < currentLayer.KerDimCount)
						{
							if (kerDimIndex == 1)
							{
								posBuffer[1]++;
							}

							if (((posBuffer[kerDimIndex] - stridePosBuffer[kerDimIndex])) > currentLayer.KerDim[kerDimIndex])
							{
								posBuffer[kerDimIndex]++;
								posBuffer[kerDimIndex - 1] = 0 + stridePosBuffer[kerDimIndex - 1];
								kerDimMult *= currentLayer.KerDim[kerDimIndex];

							}

							else
							{
								break;
							}
							kerDimIndex++;
						}



					}


					stridePosBuffer[0] += currentLayer.Stride[0];
					unsigned kerDimIndex = 0;
					while (kerDimIndex < currentLayer.KerDimCount - 1)
					{

						if ((stridePosBuffer[kerDimIndex] + (currentLayer.KerDim[kerDimIndex] - 1)) >= (prevLayer.LayerDim[kerDimIndex]))
						{

							stridePosBuffer[kerDimIndex] = 0;
							stridePosBuffer[kerDimIndex + 1] += currentLayer.Stride[kerDimIndex + 1];


						}

						posBuffer[kerDimIndex] = 0 + stridePosBuffer[kerDimIndex];
						posBuffer[kerDimIndex + 1] = 0 + stridePosBuffer[kerDimIndex + 1];

						kerDimIndex++;
					}







					

					float z = max;
					Z[featureMapIndex] = z;
					float a = n->m_Functions.NeuronFunctions[n->m_LayerLayoutPosition - 1].f(z);
					A[featureMapIndex] = a;


					featureMapIndex++;
				}



				featureMap++;
			}


			delete[] posBuffer;
			delete[] stridePosBuffer;


		}

		void PoolingLayerBackpropegateZ(NetworkPrototype* n)
		{
			LayerLayout currentLayer = n->m_LayerLayout[n->m_LayerLayoutPosition];
			LayerLayout latterLayer = n->m_LayerLayout[n->m_LayerLayoutPosition + 1];

			//Used with the "TensorOverlay" class to navigate the node array of currentLayer as an N-dimentional box.
			unsigned* posBuffer = new unsigned[latterLayer.KerDimCount];
			unsigned* stridePosBuffer = new unsigned[latterLayer.KerDimCount];

			//Set the delta weights array
			memset(currentLayer.dZ, 0.0, currentLayer.ZCount * sizeof(float));


			//Node count of individual feature maps
			unsigned featureMapNodeCount = latterLayer.NodesCount / latterLayer.SubLayerCount;
			unsigned featureMapWeightsCount = latterLayer.WeightsCount / latterLayer.SubLayerCount;
			unsigned featureMapWeightLinesCount = featureMapWeightsCount / latterLayer.KerDim[0];
			unsigned featureMapBiasesCount = latterLayer.BiasesCount / latterLayer.SubLayerCount;

			unsigned featureMap = 0;
			while (featureMap < latterLayer.SubLayerCount)
			{

				memset(posBuffer, 0, latterLayer.KerDimCount * sizeof(unsigned));
				memset(stridePosBuffer, 0, latterLayer.KerDimCount * sizeof(unsigned));



				float* latterDZ = latterLayer.dZ + featureMap * featureMapNodeCount;
				float* latterZ = latterLayer.Z + featureMap * featureMapNodeCount;


				TensorOverlay dZ(currentLayer.dZ, currentLayer.ZCount, currentLayer.LayerDim, currentLayer.LayerDimCount);
				TensorOverlay Z(currentLayer.Z, currentLayer.ZCount, currentLayer.LayerDim, currentLayer.LayerDimCount);

				unsigned featureMapIndex = 0;
				while (featureMapIndex < featureMapNodeCount)
				{

					float latterdz = latterDZ[featureMapIndex];
					float latterz = latterZ[featureMapIndex];



					unsigned kerLine = 0;
					while (kerLine < featureMapWeightLinesCount)
					{

						float* dZLine = dZ.At(posBuffer);
						float* ZLine = Z.At(posBuffer);
						unsigned kerLineIndex = 0;
						while (kerLineIndex < latterLayer.KerDim[0])
						{
							if (ZLine[kerLineIndex] == latterz)
							{
								float dAdZ = n->m_Functions.NeuronFunctionsDerivatives[n->m_LayerLayoutPosition - 1].f(ZLine[kerLineIndex]);

								dZLine[kerLineIndex] += dAdZ * latterdz;
							}
							else
							{
								dZLine[kerLineIndex] = 0;
							}
							

							kerLineIndex++;
						}

						kerLine++;


						unsigned kerDimMult = 1;
						unsigned kerDimIndex = 1;
						while (kerDimIndex < latterLayer.KerDimCount)
						{
							if (kerDimIndex == 1)
							{
								posBuffer[1]++;
							}

							if (((posBuffer[kerDimIndex] - stridePosBuffer[kerDimIndex])) > latterLayer.KerDim[kerDimIndex])
							{
								posBuffer[kerDimIndex]++;
								posBuffer[kerDimIndex - 1] = 0 + stridePosBuffer[kerDimIndex - 1];
								kerDimMult *= latterLayer.KerDim[kerDimIndex];

							}

							else
							{
								break;
							}
							kerDimIndex++;
						}



					}


					stridePosBuffer[0] += latterLayer.Stride[0];
					unsigned kerDimIndex = 0;
					while (kerDimIndex < latterLayer.KerDimCount - 1)
					{

						if ((stridePosBuffer[kerDimIndex] + (latterLayer.KerDim[kerDimIndex] - 1)) >= (currentLayer.LayerDim[kerDimIndex]))
						{

							stridePosBuffer[kerDimIndex] = 0;
							stridePosBuffer[kerDimIndex + 1] += latterLayer.Stride[kerDimIndex + 1];



						}

						posBuffer[kerDimIndex] = 0 + stridePosBuffer[kerDimIndex];
						posBuffer[kerDimIndex + 1] = 0 + stridePosBuffer[kerDimIndex + 1];

						kerDimIndex++;
					}












					featureMapIndex++;
				}



				featureMap++;
			}


			delete[] posBuffer;
			delete[] stridePosBuffer;
		}

		void PoolingLayerBackpropegateBW(NetworkPrototype* n)
		{

		}

	}



	namespace CostFunctions
	{
		void EmptyCostFunction(NetworkPrototype* n)
		{

		}


		void CrossEntropy(NetworkPrototype* n)
		{

			//REMINDER: Weights, Biases and Z - arrays corresponding to layer n, are all saved on the n-1 spot in the m_WorkloadLayout array, but not in the m_LayerLayout array.

			unsigned layerIndex = 0;
			while (layerIndex < n->m_OutputBufferCount)
			{

				float a = n->m_OutputBuffer[layerIndex];
				float y = n->m_TargetBuffer[layerIndex];


				float cost = Math::CrossEntropy(a, y);


				n->m_CostBuffer += cost;



				layerIndex++;
			}
		}

		void CrossEntropyDerivative(NetworkPrototype* n)
		{

			unsigned layerIndex = 0;
			while (layerIndex < n->m_OutputBufferCount)
			{

				float z = n->m_LayerLayout[n->m_LayerLayoutCount - 1].Z[layerIndex];
				float a = n->m_OutputBuffer[layerIndex];
				float y = n->m_TargetBuffer[layerIndex];

				float dz = Math::CrossEntropyCostDerivative(z, a, y);

				n->m_LayerLayout[n->m_LayerLayoutCount - 1].dZ[layerIndex] = dz;


				layerIndex++;
			}

		}
	}


	namespace RegularizationFunctions
	{
		void L2Regularization(NetworkPrototype* n)
		{


			//REMINDER: Weights, BViases and Z - arrays corresponding to layer n, are all saved on the n-1 spot in the m_WorkloadLayout array, but not in the m_LayerLayout array.

			//Always remember that the 0-th layer doesnt have any weights and biases.

			unsigned layerPos = n->m_LayerLayoutPosition;
			LayerLayout currentLayer = n->m_LayerLayout[layerPos];

			unsigned index = 0;
			while (index < currentLayer.NodesCount)
			{




				currentLayer.TempWeights[index] *= (1 - (currentLayer.LearningRate * currentLayer.RegularizationConstant / ((float)n->m_Data->TrainingCount)));

				auto temp = currentLayer.TempWeights[index];

				index++;


			}
		}
	}

	namespace TrainingFunctions 
	{

		


		void GradientDecent(NetworkPrototype* n)
		{

			//REMINDER: Weights, Biases and Z - arrays corresponding to layer n, are all saved on the n-1 spot in the m_WorkloadLayout array, but not in the m_LayerLayout array.

			//Always remember that the 0-th layer doesnt have any weights and biases, which means that the amount of layers you need to count is: m_layerLayoutCount-1

			unsigned layerPos = n->m_LayerLayoutPosition;
			LayerLayout currentLayer = n->m_LayerLayout[layerPos];



			
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

}


