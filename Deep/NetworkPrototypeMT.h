#pragma once
#include "NeuralNetworkCustomVariables.h"


class std::thread;


namespace TNNT
{
	class NetworkPrototypeMT
	{

	public:

		//Order A, Weights, Biases, Z, dZ, dWeights, dBiases, WeightsBuffer, BiasesBuffer 
		float* m_NetworkFixedData;
		unsigned m_NetworkFixedDataCount;


		//Note: the m_...Count variabels treat the channels as if they've been flattend, m_LayerLayout does not. So if the input layer rperesented a 5x5 RGB image:
		//m_LayerLayout[0].NodesCount would be 25, but when the nodes of the entire network are summed up into m_ACount, the inputlayer counts for: 5x5x3 = 75.
		LayerLayout* m_LayerLayout;
		unsigned m_LayerLayoutCount;

		FunctionsLayoutMT m_Functions;

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


		float* m_CostBuffer;
		unsigned* m_GuessBuffer;

		//Note: Currently having slave-thread 0 take care of this.
		unsigned m_LayerLayoutPosition;

		unsigned* m_Indices = nullptr;

		//Temporary ownership. The network is not responsible for this pointer.
		DataSet* m_Data;

		HyperParameters m_HyperParameters;

		// 0: Training, 1: Cost, 2: Success rate
		float m_LastTime[3];


		//Multi Threading
		std::thread* m_SlaveThreads;
		unsigned m_SlaveThreadCount;
		
		//Does not treat the layer divisions as flattened, but WholeWeights and WholeBiases need all the weights and biases respectively.
		WorkloadLayout m_WorkloadLayout;

		//Synching
		bool* m_Locks;
		bool* m_SlaveFlags;
		unsigned m_MasterControlPoint = 0;


		

		


	public:

		NetworkPrototypeMT(LayerLayout* layerLayout, FunctionsLayoutMT& functions, unsigned layoutCount, unsigned slaveThreadCount, bool randomizeWeightsAndBiases = true);
		~NetworkPrototypeMT();

		float CheckSuccessRate();
		float CheckCost();

		void Train(DataSet* data, HyperParameters& params);

		unsigned Check(float* input);



	public:


		//Network helpers:

		void SetTempToBiases(unsigned thread);
		void SetBiasesToTemp(unsigned thread);

		void SetTempToWeights(unsigned thread);
		void SetWeightsToTemp(unsigned thread);

		void SetTempToBiasesAndWeights(unsigned thread);


		
		
		

		void SetData(DataSet* data);
		void ResetIndices(unsigned thread);

		void SetHyperParameters(HyperParameters& params);

		void SetInput(float* input, unsigned thread);
		void SetTarget(float* target, unsigned thread);

		void ThreadWorkloadDivider(unsigned& start, unsigned& stop, unsigned workCount, unsigned thread);
		void SpinLock(unsigned thread);
		void SlaveControlStation(unsigned position);
		void WaitForSlaves();

		void IterateOverNodes();

		//Actual network mechanisms

		void FeedForward(unsigned thread);
		void Backpropegate(unsigned thread);
		
		void Regularization(unsigned thread);
		void Train(unsigned thread);
		
		void TrainOnSet(unsigned batchCount, unsigned batch, unsigned thread);

		void TrainSlaveFunction(unsigned thread);
		void TrainMasterFunction();

		// Performance evaluation

		void CheckCostSlaveFunction(unsigned thread);
		float CheckCostMasterFunction();

		void CheckSuccessRateSlaveFunction(unsigned thread);
		float CheckSuccessRateMasterFunction();

		void CheckSlaveFunction(float* input, unsigned thread);
		unsigned CheckMasterFunction(float* input);
		


	};
}


