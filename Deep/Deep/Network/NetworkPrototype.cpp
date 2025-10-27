#include "../pch.h"
#include "NetworkPrototype.h"

//#ifdef FullyConnectedLayerDef
#include "LayerTypes/FullyConnectedLayer.h"
//#endif
#include "LayerTypes/InputLayer.h"
#include "TrainingFunctions/CostFunctions.h"
#include "TrainingFunctions/GradientDecent.h"

//
//namespace TNNT
//{
//	//Constructors And destructor
//
//
//
//
//
//
//
//
//
//
//
//
//
//
//
//
//
//
//
//	void NetworkPrototype::Regularization()
//	{
//
//
//
//
//		unsigned layoutIndex = 1;
//
//		while (layoutIndex < m_LayerLayoutCount)
//		{
//
//
//			m_LayerLayoutPointer->RegularizationFunctions(this);
//
//
//
//
//
//
//			m_LayerLayoutPosition++;
//			m_LayerLayoutPointer = m_LayerLayoutPointer->Next();
//
//
//
//			layoutIndex++;
//
//		}
//	}
//
//	void NetworkPrototype::Train()
//	{
//
//
//
//		unsigned layoutIndex = 1;
//
//		while (layoutIndex < m_LayerLayoutCount)
//		{
//
//
//			m_LayerLayoutPointer->TrainingFunctions(this);
//
//
//
//
//
//
//			m_LayerLayoutPosition++;
//			m_LayerLayoutPointer = m_LayerLayoutPointer->Next();
//
//
//
//			layoutIndex++;
//
//		}
//	}
//
//
//
//	void NetworkPrototype::TrainOnSet(unsigned batchCount, unsigned batch)
//	{
//
//
//		Regularization();
//
//
//
//
//
//
//		unsigned exampleIndex = 0;
//		while (exampleIndex < batchCount)
//		{
//
//			unsigned index = m_Indices[exampleIndex + batch * m_HyperParameters.BatchCount];
//			SetInput(&(m_Data->TrainingInputs[index * m_InputBufferCount]));
//			SetTarget(&(m_Data->TraningTargets[index * m_OutputBufferCount]));
//
//
//			FeedForward();
//			Backpropegate();
//
//			Train();
//
//
//
//
//
//			exampleIndex++;
//
//
//		}
//
//
//		SetBiasesToTemp();
//		SetWeightsToTemp();
//
//	}
//
//
//	void NetworkPrototype::TrainMasterFunction()
//	{
//
//		//Timer start
//		auto start = std::chrono::high_resolution_clock::now();
//
//
//
//
//		const unsigned batchNum = m_Data->TrainingCount / m_HyperParameters.BatchCount;
//		const unsigned remainingBatch = m_Data->TrainingCount % m_HyperParameters.BatchCount;
//
//		std::mt19937 mt;
//
//
//		unsigned epochNum = 0;
//		while (epochNum < m_HyperParameters.Epochs)
//		{
//
//			unsigned randomIndexPos = 0;
//			unsigned randomIndexCount = m_Data->TrainingCount;
//
//
//			unsigned batch = 0;
//			while (batch < batchNum)
//			{
//				unsigned batchIndex = 0;
//				while (batchIndex < m_HyperParameters.BatchCount)
//				{
//
//					unsigned randomIndex = (mt() % randomIndexCount) + randomIndexPos;
//
//					unsigned epochRandomIndex = m_Indices[randomIndex];
//					m_Indices[randomIndex] = m_Indices[randomIndexPos];
//					m_Indices[randomIndexPos] = epochRandomIndex;
//
//
//					randomIndexPos++;
//					randomIndexCount--;
//
//					batchIndex++;
//				}
//
//				TrainOnSet(m_HyperParameters.BatchCount, batch);
//
//				batch++;
//			}
//
//
//			if (remainingBatch > 0)
//			{
//				unsigned batchIndex = 0;
//				while (batchIndex < remainingBatch)
//				{
//
//
//					unsigned randomIndex = (mt() % randomIndexCount) + randomIndexPos;
//
//					unsigned epochRandomIndex = m_Indices[randomIndex];
//					m_Indices[randomIndex] = m_Indices[randomIndexPos];
//					m_Indices[randomIndexPos] = epochRandomIndex;
//
//
//
//					randomIndexPos++;
//					randomIndexCount--;
//					batchIndex++;
//
//				}
//				unsigned tempBatchCount = m_HyperParameters.BatchCount;
//				m_HyperParameters.BatchCount = remainingBatch;
//				TrainOnSet(remainingBatch, batch);
//				m_HyperParameters.BatchCount = tempBatchCount;
//			}
//
//
//			epochNum++;
//		}
//
//
//	}
//
//
//	float NetworkPrototype::CheckCostMasterFunction()
//	{
//
//		auto start = std::chrono::high_resolution_clock::now();
//
//
//		m_CostBuffer = 0;
//
//		unsigned checkIndex = 0;
//		while (checkIndex < m_Data->TestCount)
//		{
//
//			SetInput(&m_Data->TestInputs[checkIndex * m_InputBufferCount]);
//			SetTarget(&m_Data->TestTargets[checkIndex * m_OutputBufferCount]);
//
//
//			FeedForward();
//
//			m_CostFunction(this);
//
//
//			checkIndex++;
//		}
//
//
//
//
//
//
//		return  m_CostBuffer / ((float)m_Data->TestCount);
//
//	}
//
//	float NetworkPrototype::CheckSuccessRateMasterFunction()
//	{
//		auto start = std::chrono::high_resolution_clock::now();
//
//
//
//		float score = 0.0f;
//
//		unsigned checkIndex = 0;
//		while (checkIndex < m_Data->TestCount)
//		{
//
//			SetInput(&m_Data->TestInputs[checkIndex * m_InputBufferCount]);
//			FeedForward();
//
//			int championItterator = -1;
//			float champion = 0;
//			unsigned outputIndex = 0;
//			while (outputIndex < m_OutputBufferCount)
//			{
//
//				if (m_OutputBuffer[outputIndex] >= champion)
//				{
//					champion = m_OutputBuffer[outputIndex];
//					championItterator = outputIndex;
//				}
//
//
//				outputIndex++;
//			}
//
//			if (m_Data->TestTargets[m_OutputBufferCount * checkIndex + championItterator] == 1)
//			{
//				score += 1.0f;
//			}
//			checkIndex++;
//		}
//
//		float rate = score / ((float)m_Data->TestCount);
//
//
//
//		return rate;
//
//	}
//
//	unsigned NetworkPrototype::CheckMasterFunction(float* input)
//	{
//		SetInput(input);
//		FeedForward();
//
//		int champIndex = -1;
//		float champ = 0;
//
//		unsigned index = 0;
//		while (index < m_OutputBufferCount)
//		{
//			if (m_OutputBuffer[index] > champ)
//			{
//				champ = m_OutputBuffer[index];
//				champIndex = index;
//			}
//			index++;
//		}
//
//		//This is not allowed.
//		assert(champIndex != -1);
//
//
//		return champIndex;
//	}
//
//
//
//
//
//
//
	//NEW STUFF ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


namespace TNNT{


#define Layer1NodeCount 784
#define Layer2NodeCount 100
#define Layer3NodeCount 30
#define Layer4NodeCount 10

	InputLayerParams il;
	FullyConnectedHyperParams l1;
	FullyConnectedHyperParams l2;
	FullyConnectedHyperParams l3;


	float* NetwrokStorage = new float[
		Layer1NodeCount * 4 + Layer1NodeCount * Layer2NodeCount * 4 + Layer2NodeCount * 3 + Layer2NodeCount * 4 +
			Layer2NodeCount * 4 + Layer2NodeCount * Layer3NodeCount * 4 + Layer3NodeCount * 3 + Layer3NodeCount * 4 +
			Layer3NodeCount * 4 + Layer3NodeCount * Layer4NodeCount * 4 + Layer4NodeCount * 3 + Layer4NodeCount * 4
			 ];

	unsigned* TrainingIndices = new unsigned[50000];

	TrainingMetaData metadata;
	
	


	void SetInput(float* input)
	{
		memcpy(NetwrokStorage, input, InputNodesCount*sizeof(float));

		//PrintImg(l1.inputZ);
	}

	void SetTarget(float* target)
	{
	}


	void SetupNetwork(float* trainingData, float* trainingTargets, unsigned trainingCount)
	{


		float* layerStorage = NetwrokStorage;

		metadata.TrainingIndices = TrainingIndices;
		for (unsigned i = 0; i < trainingCount; i++)
		{
			metadata.TrainingIndices[i] = i;
		}
		metadata.TrainingCount = trainingCount;
		metadata.TrainingDataInputs = trainingData;
		metadata.TrainingDataTargets = trainingTargets;



		layerStorage = InputLayerSetup(il, layerStorage, Layer1NodeCount);

		layerStorage = FullyConnectedSetup(l1, layerStorage, Layer1NodeCount, Layer2NodeCount, Layer1NodeCount * Layer2NodeCount, Layer2NodeCount);

		layerStorage = FullyConnectedSetup(l2, layerStorage, Layer2NodeCount, Layer3NodeCount, Layer2NodeCount * Layer3NodeCount, Layer3NodeCount);

		layerStorage = FullyConnectedSetup(l3, layerStorage, Layer3NodeCount, Layer4NodeCount, Layer3NodeCount * Layer4NodeCount, Layer4NodeCount);



	}

	void FeedForwardPass()
	{
		InputLayerTransform(il);
		FullyConnectedFeedForward(l1);
		FullyConnectedFeedForward(l2);
		FullyConnectedFeedForward(l3);
	}

	void BackpropegationPass(float* target)
	{

		CrossEntropyDerivative(l3.outputA, l3.outputZ, l3.outputDZ, target, l3.outputNodesCount);

		FullyConnectedBackpropegateZ(l3);
		FullyConnectedBackpropegateBW(l3);

		FullyConnectedBackpropegateZ(l2);
		FullyConnectedBackpropegateBW(l2);

		FullyConnectedBackpropegateBW(l1);
	}

	void PreParameterAdjustment(TrainingMetaData m )
	{
		L2Regularization(l1.tempWeights, l1.WeightsCount, m.TrainingCount, m.LearningRate, m.RegConstant);
		L2Regularization(l2.tempWeights, l2.WeightsCount, m.TrainingCount, m.LearningRate, m.RegConstant);
		L2Regularization(l3.tempWeights, l3.WeightsCount, m.TrainingCount, m.LearningRate, m.RegConstant);
	}

	void ParameterAdjustment(TrainingMetaData m, unsigned actualBatchSize)
	{
		GradientDecent(l1.tempWeights, l1.DWeights, l1.WeightsCount, actualBatchSize, m.LearningRate);
		GradientDecent(l2.tempWeights, l2.DWeights, l2.WeightsCount, actualBatchSize, m.LearningRate);
		GradientDecent(l3.tempWeights, l3.DWeights, l3.WeightsCount, actualBatchSize, m.LearningRate);

		GradientDecent(l1.tempBiases, l1.DBiases, l1.BiasesCount, actualBatchSize, m.LearningRate);
		GradientDecent(l2.tempBiases, l2.DBiases, l2.BiasesCount, actualBatchSize, m.LearningRate);
		GradientDecent(l3.tempBiases, l3.DBiases, l3.BiasesCount, actualBatchSize, m.LearningRate);
	}

	void PostParameterAdjustment()
	{
		FullyConnectedSetWeightsToTemp(l1);
		FullyConnectedSetBiasesToTemp(l1);
		FullyConnectedSetWeightsToTemp(l2);
		FullyConnectedSetBiasesToTemp(l2);
		FullyConnectedSetWeightsToTemp(l3);
		FullyConnectedSetBiasesToTemp(l3);
	}

	void TrainOnSet(TrainingMetaData m, unsigned actualBatchSize)
	{

		PreParameterAdjustment(m);

		unsigned exampleIndex = 0;
		while (exampleIndex < actualBatchSize)
		{

			unsigned index = m.TrainingIndices[exampleIndex];

			SetInput(&(m.TrainingDataInputs[index * m.InputSize]));
			FeedForwardPass();
			BackpropegationPass(&m.TrainingDataTargets[index * m.TargetSize]);

			ParameterAdjustment(m, actualBatchSize);


			exampleIndex++;


		}

		PostParameterAdjustment();

	}


	void TrainingLoop()
	{
		
		TrainingMetaData& m = metadata;



		const unsigned batchNum = m.TrainingCount / m.BatchSize;
		const unsigned remainingBatch = m.TrainingCount % m.BatchSize;

		std::mt19937 mt;


		unsigned epochNum = 0;
		while (epochNum < m.Epochs)
		{

			ShuffleInt(m.TrainingIndices, mt, m.TrainingCount);

			unsigned randomIndexPos = 0;
			unsigned randomIndexCount = m.TrainingCount;


			unsigned batch = 0;
			while (batch < batchNum)
			{


				TrainOnSet(m, m.BatchSize);

				batch++;
			}


			if (remainingBatch > 0)
			{


				TrainOnSet(m, remainingBatch);

			}

			metadata.LearningRate *= 0.9;


			epochNum++;
		}

	}

	void TestNetwork(float* testData, float* testTargets, unsigned testCount)
	{

		float cost = 0;
		float successrate = 0;

		unsigned testIndex = 0;
		while (testIndex < testCount)
		{
			
			SetInput(&testData[testIndex *InputNodesCount]);
			FeedForwardPass();

			cost += CrossEntropy(l3.outputA, &testTargets[testIndex * OutputNodesCount], OutputNodesCount);
			


			//pr("Guess");
			//PArr<float>(l3.outputA , OutputNodesCount);
			//pr("Target");
			//PArr<float>(&testTargets[testIndex * OutputNodesCount], OutputNodesCount);


			unsigned champindex = OutputNodesCount;
			float champ = 0;
			for (unsigned i = 0; i < OutputNodesCount; i++)
			{
				if (champ < l3.outputA[i])
				{
					champ = l3.outputA[i];
					champindex = i;
				}
			}

			if (champindex != OutputNodesCount)
			{

				if (testTargets[testIndex * OutputNodesCount + champindex] > 0.5f)
				{
					successrate++;
				}
			}


			testIndex++;
		}
		cost = cost / testCount;
		successrate = successrate / testCount;

		pr("Cost: " << cost);
		pr("Successrate: " << successrate);

	}



}


