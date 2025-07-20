#pragma once
#include "NeuralNetworkCustomVariables.h"
#include "Layer.h"


namespace TNNT {

	class NetworkPrototype
	{
	public:

		//Order A, Weights, Biases, Z, dZ, dWeights, dBiases, WeightsBuffer, BiasesBuffer 
		char* m_NetworkFixedData;


		//Note: the m_...Count variabels treat the channels as if they've been flattend, m_LayerLayout does not. So if the input layer rperesented a 5x5 RGB image:
		//m_LayerLayout[0].NodesCount would be 25, but when the nodes of the entire network are summed up into m_ACount, the inputlayer counts for: 5x5x3 = 75. 
		//This may or may not be a lie; dont trust everything you read on the computer.
		//Make sure that the sizeof(LayerHeader*) * layoutCount bytes of layerLayout (in the constructor), after the first 16 (or maybe 32 in the future),
		//are reserved for the creation of m_LayerLayout
		char* m_LayerLayoutBuffer;
		LayerHeader* m_LayerLayoutPointer;
		LayerHeader** m_LayerLayout;

		NetworkRelayFunction m_CostFunction;
		

		float* m_A;
		float* m_InputBuffer;
		float* m_OutputBuffer;

		float* m_Weights;
		float* m_Biases;

		float* m_Z;
		float* m_DeltaZ;







		





		float* m_WeightsTranspose; 

		float* m_TempWeights;
		float* m_TempBiases;

		float* m_DeltaWeights;
		float* m_DeltaBiases;

		float* m_TargetBuffer;


		
		float m_CostBuffer;


		unsigned m_NetworkFixedDataSize;

		unsigned m_LayerLayoutCount;

		unsigned m_ACount;
		unsigned m_ZCount;

		unsigned m_InputBufferCount;
		unsigned m_OutputBufferCount;

		unsigned m_WeightsCount;
		unsigned m_BiasesCount;



		
		HyperParameters m_HyperParameters;
		
		unsigned* m_Indices = nullptr;

		unsigned m_LayerLayoutPosition;
		
		DataSet* m_Data = nullptr;

		// 0: Training, 1: Cost, 2: Success rate
		float m_LastTime[3];


	public:

		NetworkPrototype(char* layerLayout, unsigned layoutCount , bool randomizeWeightsAndBiases = true);
		~NetworkPrototype();

		float CheckSuccessRate();
		float CheckCost();

		void SaveParams();
		void LoadParams();

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

}