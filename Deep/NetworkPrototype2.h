#pragma once
#include "NeuralNetworkCustomVariables.h"
#include "Layer.h"


namespace TNNT {

	class NetworkPrototype2
	{
	public:

		//Order A, Weights, Biases, Z, dZ, dWeights, dBiases, WeightsBuffer, BiasesBuffer 
		float* m_NetworkFixedData;
		unsigned m_NetworkFixedDataCount;

		//Note: the m_...Count variabels treat the channels as if they've been flattend, m_LayerLayout does not. So if the input layer rperesented a 5x5 RGB image:
		//m_LayerLayout[0].NodesCount would be 25, but when the nodes of the entire network are summed up into m_ACount, the inputlayer counts for: 5x5x3 = 75.
		FCLayer* m_LayerLayout;
		unsigned m_LayerLayoutCount;


		float* m_A;
		unsigned m_ACount;


		float* m_Z;
		float* m_DeltaZ;
		unsigned m_ZCount;


		float* m_InputBuffer;
		unsigned m_InputBufferCount;

		float* m_OutputBuffer;
		float* m_TargetBuffer;
		unsigned m_OutputBufferCount;


		float* m_Weights;
		float* m_Biases;

		float* m_TempWeights;
		float* m_TempBiases;

		float* m_DeltaWeights;
		float* m_DeltaBiases;

		unsigned m_WeightsCount;
		unsigned m_BiasesCount;

		
		float m_CostBuffer;
		void (*m_CostFunction)(NetworkPrototype2* n);
		void (*m_CostFunctionDerivative)(NetworkPrototype2* n);
		
		
		HyperParameters m_HyperParameters;
		
		unsigned* m_Indices = nullptr;

		unsigned m_LayerLayoutPosition;
		
		DataSet* m_Data = nullptr;

		// 0: Training, 1: Cost, 2: Success rate
		float m_LastTime[3];


	public:

		NetworkPrototype2(FCLayer* layerLayout, void (*costFunction)(NetworkPrototype2*), void (*costFunctionDerivative)(NetworkPrototype2*), unsigned layoutCount , bool randomizeWeightsAndBiases = true);
		~NetworkPrototype2();

		float CheckSuccessRate();
		float CheckCost();

		void Train(DataSet* data, HyperParameters& params);

		unsigned Check(float* input);



	public:


		//Network helpers:

		void SetBiasesToTemp();
		void SetTempToBiases();

		void SetWeightsToTemp();
		void SetTempToWeights();

		void SetData(DataSet* data);
		void SetHyperParameters(HyperParameters& params);

		void SetInput(const float* input);
		void SetTarget(const float* target);

		//Actual network mechanisms

		void FeedForward();
		void FeedForwardCost();
		void Backpropegate();
	
		void Regularization();
		void Train();

		void TrainOnSet(unsigned batchCount, unsigned batch);

		void TrainMasterFunction();
		
		// Performance evaluation

		float CheckCostMasterFunction();
		float CheckSuccessRateMasterFunction();
		unsigned CheckMasterFunction(float* input);



	};


	void CrossEntropyTest(NetworkPrototype2* n);
	void CrossEntropyDerivativeTest(NetworkPrototype2* n);
}