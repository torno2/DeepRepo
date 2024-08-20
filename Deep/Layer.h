#pragma once

#include "NeuralNetworkCustomVariables.h"
namespace TNNT
{

	class NetworkPrototype2;

	class FCLayer
	{
	public:
		float* A			=nullptr;
		float* Z			=nullptr;
		float* Weights		=nullptr;
		float* Biases		=nullptr;

		float* dZ			=nullptr;
		float* dWeights		=nullptr;
		float* dBiases		=nullptr;
		float* TempWeights	=nullptr;
		float* TempBiases	=nullptr;


		unsigned NodesCount = 30;
		unsigned ZCount = 30;
		unsigned BiasesCount = 30;
		unsigned WeightsCount = 30*30;

		float LearningRate = 0.01f;
		float RegularizationConstant = 0.001f;

		float NeuronFunction(float z);
		float NeuronFunctionDerivative(float z);
		void FeedForward(NetworkPrototype2* n);
		void BackPropegateZ(NetworkPrototype2* n);
		void BackPropegateBW(NetworkPrototype2* n);

		void Regularize(NetworkPrototype2* n);
		void Train(NetworkPrototype2* n);
	};



}

