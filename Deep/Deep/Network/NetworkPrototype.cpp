//#include "../pch.h"
//#include "NetworkPrototype.h"
//
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
//	//NEW STUFF ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
//
//
//	
//
//
//
//	void TrainingLoop()
//	{
//		FullyConnectedFeedForward(l1);
//		FullyConnectedFeedForward(l2);
//		FullyConnectedFeedForward(l3);
//
//		FullyConnectedBackpropegateZ(l3);
//		FullyConnectedBackpropegateBW(l3);
//
//		FullyConnectedBackpropegateZ(l2);
//		FullyConnectedBackpropegateBW(l2);
//
//		FullyConnectedBackpropegateBW(l1);
//
//
//	}
//
//}