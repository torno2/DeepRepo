#include "pch.h"
#include "NetworkPrototype2.h"
#include "LayerFunctions.h"

namespace TNNT
{
	//Constructors And destructor

	NetworkPrototype2::NetworkPrototype2(FCLayer* layerLayout, void (*costFunction)(NetworkPrototype2*), void (*costFunctionDerivative)(NetworkPrototype2*), unsigned layoutCount, bool randomizeWeightsAndBiases)
		:m_LayerLayout(layerLayout), m_LayerLayoutCount(layoutCount)
	{


		// 1 layer no network makes; need at least 2
		assert(m_LayerLayoutCount >= 2);

		// A layer of zero nodes would mean you have two separate networks (or a network with 1 less layer, if the input/output layer is missing), 
		// and a layer with a negative numbers of nodes is something I don't want to think about.

		//These are gonna get reused a lot in the constructor.
		unsigned layoutIndex = 0;

		//Getting the LayerLayout ready START

		//m_LayerLayout = new FCLayer[m_LayerLayoutCount];

		//layoutIndex = 0;
		//while (layoutIndex < m_LayerLayoutCount)
		//{
		//	m_LayerLayout[layoutIndex] = layerLayout[layoutIndex];

		//	layoutIndex++;
		//}

		m_CostFunction = costFunction;
		m_CostFunctionDerivative = costFunctionDerivative;

		//Getting the LayerLayout ready STOP




		// DETERMENING THE COUNT OF MOST ARRAYS AND CALCULATING OTHER IMPORTANT INTEGERS START



		//Keep in mind that there aren't supposed to be any biases or weights in the 0th layer, so their count for that layer should both be 0. (This may or may not be wrong)
		unsigned nodesTotal = 0;
		unsigned zTotal = 0;
		unsigned biasTotal = 0;
		unsigned weightTotal = 0;


		layoutIndex = 0;
		while (layoutIndex < m_LayerLayoutCount)
		{
			// A layer of zero nodes would mean you have two separate networks (or a network with 1 less layer, if the input/output layer is missing), 
			// and a layer with a negative numbers of nodes is something I don't want to think about.
			assert(m_LayerLayout[layoutIndex].NodesCount > 0);

			nodesTotal += m_LayerLayout[layoutIndex].NodesCount;
			zTotal += m_LayerLayout[layoutIndex].ZCount;
			biasTotal += m_LayerLayout[layoutIndex].BiasesCount;
			weightTotal += m_LayerLayout[layoutIndex].WeightsCount;

			layoutIndex++;
		}

		m_ACount = nodesTotal;
		m_ZCount = zTotal;
		m_BiasesCount = biasTotal;
		m_WeightsCount = weightTotal;

		m_InputBufferCount = m_LayerLayout[0].NodesCount;
		m_OutputBufferCount = m_LayerLayout[m_LayerLayoutCount - 1].NodesCount;

		

		// DETERMENING THE COUNT OF MOST ARRAYS AND CALCULATING OTHER IMPORTANT INTEGERS STOP



		//MEMORY ALLOCATION AND POINTER SETUP START

		//Order: A, Weights, Biases, Z, dZ, WeightsBuffer, BiasesBuffer, dWeights, dBiases,  Target
		m_NetworkFixedDataCount = (m_ACount)+3 * (m_WeightsCount)+3 * (m_BiasesCount)+2 * (m_ZCount)+(m_OutputBufferCount);
		m_NetworkFixedData = new float[m_NetworkFixedDataCount];

		//NETWORK STRUCTURE
		m_A = m_NetworkFixedData;
		m_InputBuffer = m_A;
		m_OutputBuffer = m_A + m_ACount - m_OutputBufferCount;

		m_Weights = m_A + m_ACount;
		m_Biases = m_Weights + m_WeightsCount;

		m_Z = m_Biases + m_BiasesCount;
		m_DeltaZ = m_Z + m_ZCount;

		m_TempWeights = m_DeltaZ + m_ZCount;
		m_TempBiases = m_TempWeights + m_WeightsCount;

		m_DeltaWeights = m_TempBiases + m_BiasesCount;
		m_DeltaBiases = m_DeltaWeights + m_WeightsCount;

		//EVALUATION BUFFERS
		m_TargetBuffer = m_DeltaBiases + m_BiasesCount;




		// Setting up pointers in the layer layout.
		{
			unsigned adjustA = 0;
			unsigned aAdjustZ = 0;
			unsigned aAdjustWeights = 0;
			unsigned aAdjustBiases = 0;
			layoutIndex = 0;
			while (layoutIndex < m_LayerLayoutCount)
			{

				m_LayerLayout[layoutIndex].A = m_A + adjustA;

				adjustA += m_LayerLayout[layoutIndex].NodesCount;


				m_LayerLayout[layoutIndex].Z = m_Z + aAdjustZ;
				m_LayerLayout[layoutIndex].dZ = m_DeltaZ + aAdjustZ;

				aAdjustZ += m_LayerLayout[layoutIndex].ZCount;


				m_LayerLayout[layoutIndex].Weights = m_Weights + aAdjustWeights;
				m_LayerLayout[layoutIndex].dWeights = m_DeltaWeights + aAdjustWeights;
				m_LayerLayout[layoutIndex].TempWeights = m_TempWeights + aAdjustWeights;

				aAdjustWeights += m_LayerLayout[layoutIndex].WeightsCount;


				m_LayerLayout[layoutIndex].Biases = m_Biases + aAdjustBiases;
				m_LayerLayout[layoutIndex].dBiases = m_DeltaBiases + aAdjustBiases;
				m_LayerLayout[layoutIndex].TempBiases = m_TempBiases + aAdjustBiases;

				aAdjustBiases += m_LayerLayout[layoutIndex].BiasesCount;



				layoutIndex++;
			}
		}


		//MEMORY ALLOCATION AND POINTER SETUP STOP






		//ENSURING THAT CERTAIN INTEGER AND FLOAT ARRAYS HAVE ACCEPTABLE INITIAL VALUES START







		//WEIGHTS AND BIASES SETUP START
		if (randomizeWeightsAndBiases)
		{
			//For randomly initializing the weights and biases
			std::default_random_engine generator;
			std::normal_distribution<float> distribution(0.0f, 1 / sqrt(m_LayerLayout[0].NodesCount));


			unsigned index = 0;
			while (index < m_WeightsCount)
			{


				float temp = distribution(generator);
				m_Weights[index] = temp;

				index++;
			}




			index = 0;
			while (index < m_BiasesCount)
			{

				float temp = distribution(generator);
				m_Biases[index] = temp;

				index++;
			}



		}

		else
		{
			//Sets all weights and biases to zero

			unsigned index = 0;
			while (index < m_WeightsCount)
			{



				m_Weights[index] = 0;

				index++;
			}




			index = 0;
			while (index < m_BiasesCount)
			{


				m_Biases[index] = 0;

				index++;
			}
		}

		SetTempToWeights();
		SetTempToBiases();
		//WEIGHTS AND BIASES SETUP STOP



	//ENSURING THAT CERTAIN INTEGER AND FLOAT ARRAYS HAVE ACCEPTABLE INITIAL VALUES STOP

	}

	NetworkPrototype2::~NetworkPrototype2()
	{
		//delete[] m_LayerLayout;

		delete[] m_NetworkFixedData;


		delete[] m_Indices;
	}

	


	float NetworkPrototype2::CheckSuccessRate()
	{
		return CheckSuccessRateMasterFunction( );
	}

	float NetworkPrototype2::CheckCost()
	{
		return CheckCostMasterFunction();
	}

	void NetworkPrototype2::Train(DataSet* data, HyperParameters& params)
	{
		SetData(data);
		SetHyperParameters(params);
		TrainMasterFunction();
	}

	unsigned NetworkPrototype2::Check(float* input)
	{
		return(CheckMasterFunction(input));
	}





	//Public functions


	// Private functions

	void NetworkPrototype2::SetBiasesToTemp()
	{

		memcpy(m_Biases, m_TempBiases, sizeof(float) * m_BiasesCount);
	}

	void NetworkPrototype2::SetTempToBiases()
	{

		memcpy(m_TempBiases, m_Biases, sizeof(float) * m_BiasesCount);

	}


void NetworkPrototype2::SetWeightsToTemp()	
	{	

		memcpy(m_Weights, m_TempWeights, sizeof(float) * m_WeightsCount);
		
	}

	void NetworkPrototype2::SetTempToWeights()
	{

		memcpy(m_TempWeights, m_Weights, sizeof(float) * m_WeightsCount);
	}




	void NetworkPrototype2::SetData(DataSet* data)
	{
		delete[] m_Indices;

		m_Data = data;

		m_Indices = new unsigned[m_Data->TrainingCount];



		unsigned index = 0;
		while (index < m_Data->TrainingCount)
		{
			m_Indices[index] = index;
			index++;
		}

	}



	void NetworkPrototype2::SetHyperParameters(HyperParameters& params)
	{

		m_HyperParameters = params;


	}


	void NetworkPrototype2::SetInput(const float* input)
	{

		memcpy(m_InputBuffer, input, sizeof(float) * m_InputBufferCount);

	}

	void NetworkPrototype2::SetTarget(const float* target)
	{



		memcpy(m_TargetBuffer, target, sizeof(float) * m_OutputBufferCount);


	}


	void NetworkPrototype2::FeedForward()
	{


		
		m_LayerLayoutPosition = 1;

	
		while (m_LayerLayoutPosition < m_LayerLayoutCount)
		{

			//No function for the inputlayer, which means that the function corresponding to any other layer is located at layer - 1.
			m_LayerLayout[m_LayerLayoutPosition].FeedForward(this);


			m_LayerLayoutPosition++;
		}

	}

	void NetworkPrototype2::FeedForwardCost()
	{
		m_LayerLayoutPosition = 1;


		while (m_LayerLayoutPosition < m_LayerLayoutCount)
		{

			//No function for the inputlayer, which means that the function corresponding to any other layer is located at layer - 1.
			m_LayerLayout[m_LayerLayoutPosition].FeedForward(this);


			m_LayerLayoutPosition++;
		}
	}

	void NetworkPrototype2::Backpropegate()
	{

		m_LayerLayoutPosition = m_LayerLayoutCount - 1;
		m_CostFunctionDerivative(this);


		while (2 <= m_LayerLayoutPosition)
		{
			
			m_LayerLayout[m_LayerLayoutPosition].BackPropegateBW(this);

			m_LayerLayoutPosition--;
			
			m_LayerLayout[m_LayerLayoutPosition].BackPropegateZ(this);


			
		}
		m_LayerLayout[m_LayerLayoutPosition].BackPropegateBW(this);
	}

	void NetworkPrototype2::Regularization()
	{

		
		m_LayerLayoutPosition = 1;

		while (m_LayerLayoutPosition < m_LayerLayoutCount)
		{

			
			m_LayerLayout[m_LayerLayoutPosition].Regularize(this);




			m_LayerLayoutPosition++;

		}
	}

	void NetworkPrototype2::Train()
	{
		
		m_LayerLayoutPosition = 1;


		while (m_LayerLayoutPosition < m_LayerLayoutCount)
		{


			m_LayerLayout[m_LayerLayoutPosition].Train(this);




			

			m_LayerLayoutPosition++;

		}
	}

 

	void NetworkPrototype2::TrainOnSet(unsigned batchCount , unsigned batch)
	{


		Regularization();


		

	

		unsigned exampleIndex = 0;
		while (exampleIndex < batchCount)
		{

			unsigned index = m_Indices[exampleIndex + batch * m_HyperParameters.BatchCount];
			SetInput(&(m_Data->TrainingInputs[index * m_InputBufferCount]));
			SetTarget(&(m_Data->TraningTargets[index * m_OutputBufferCount]));


			FeedForwardCost();
			Backpropegate();

			Train();


			


			exampleIndex++;


		}


		SetBiasesToTemp();
		SetWeightsToTemp();

	}


	void NetworkPrototype2::TrainMasterFunction()
	{

		//Timer start
		auto start = std::chrono::high_resolution_clock::now();




		const unsigned batchNum = m_Data->TrainingCount / m_HyperParameters.BatchCount;
		const unsigned remainingBatch = m_Data->TrainingCount % m_HyperParameters.BatchCount;

		std::mt19937 mt;
		

		unsigned epochNum = 0;
		while (epochNum < m_HyperParameters.Epochs)
		{
			
			unsigned randomIndexPos = 0;
			unsigned randomIndexCount = m_Data->TrainingCount;


			unsigned batch = 0;
			while (batch < batchNum)
			{
				unsigned batchIndex = 0;
				while (batchIndex < m_HyperParameters.BatchCount)
				{

					unsigned randomIndex = (mt() % randomIndexCount)+ randomIndexPos;

					unsigned epochRandomIndex = m_Indices[randomIndex];
					m_Indices[randomIndex] = m_Indices[randomIndexPos];
					m_Indices[randomIndexPos] = epochRandomIndex;


					randomIndexPos++;
					randomIndexCount--;

					batchIndex++;
				}

				TrainOnSet(m_HyperParameters.BatchCount, batch);

				batch++;
			}


			if (remainingBatch > 0)
			{
				unsigned batchIndex = 0;
				while (batchIndex < remainingBatch)
				{


					unsigned randomIndex = (mt() % randomIndexCount) + randomIndexPos;

					unsigned epochRandomIndex = m_Indices[randomIndex];
					m_Indices[randomIndex] = m_Indices[randomIndexPos];
					m_Indices[randomIndexPos] = epochRandomIndex;



					randomIndexPos++;
					randomIndexCount--;
					batchIndex++;

				}
				unsigned tempBatchCount = m_HyperParameters.BatchCount;
				m_HyperParameters.BatchCount = remainingBatch;
				TrainOnSet(remainingBatch, batch);
				m_HyperParameters.BatchCount = tempBatchCount;
			}


			epochNum++;
		}
		
		//TODO Remove this:
		


		


		//Timer stop
		auto stop = std::chrono::high_resolution_clock::now();
		std::chrono::duration<float>  time = stop - start;
		m_LastTime[0] = time.count();
	}

	
	float NetworkPrototype2::CheckCostMasterFunction( )
	{

		auto start = std::chrono::high_resolution_clock::now();
		

		m_CostBuffer = 0;

		unsigned checkIndex = 0;
		while (checkIndex < m_Data->TestCount)
		{

			SetInput( &m_Data->TestInputs[checkIndex * m_InputBufferCount]  );
			SetTarget(&m_Data->TestTargets[checkIndex * m_OutputBufferCount]);
			
			
			FeedForwardCost();
			
			m_CostFunction(this);
			

			checkIndex++;
		}



		auto stop = std::chrono::high_resolution_clock::now();
		std::chrono::duration<float>  time = stop - start;
		m_LastTime[1] = time.count();


	
		return  m_CostBuffer / ((float)m_Data->TestCount);

	}

	float NetworkPrototype2::CheckSuccessRateMasterFunction()
	{
		auto start = std::chrono::high_resolution_clock::now();

		

		float score = 0.0f;

		unsigned checkIndex = 0;
		while (checkIndex < m_Data->TestCount)
		{

			SetInput(&m_Data->TestInputs[checkIndex * m_InputBufferCount]);
			FeedForward();

			int championItterator = -1;
			float champion = 0;
			unsigned outputIndex = 0;
			while (outputIndex < m_OutputBufferCount)
			{
				
				if (m_OutputBuffer[outputIndex] >= champion)
				{
					champion = m_OutputBuffer[outputIndex];
					championItterator = outputIndex;
				}


				outputIndex++;
			}

			if (m_Data->TestTargets[m_OutputBufferCount * checkIndex + championItterator] == 1)
			{
				score += 1.0f;
			}
			checkIndex++;
		}

		float rate = score / ((float)m_Data->TestCount);

		auto stop = std::chrono::high_resolution_clock::now();
		std::chrono::duration<float>  time = stop - start;
		m_LastTime[2] = time.count();

		return rate;

	}

	unsigned NetworkPrototype2::CheckMasterFunction(float* input)
	{
		SetInput(input);
		FeedForward();

		int champIndex = -1;
		float champ = 0;

		unsigned index = 0;
		while (index < m_OutputBufferCount)
		{
			if (m_OutputBuffer[index] > champ)
			{
				champ = m_OutputBuffer[index];
				champIndex = index;
			}
			index++;
		}

		//This is not allowed.
		assert(champIndex != -1);


		return champIndex;
	}


	void CrossEntropyTest(NetworkPrototype2* n)
	{
		//REMINDER: Weights, Biases and Z - arrays corresponding to layer n, are all saved on the n-1 spot in the m_WorkloadLayout array, but not in the m_LayerLayout array.

		unsigned nodeCount = n->m_LayerLayout[n->m_LayerLayoutCount - 1].NodesCount;

		unsigned layerIndex = 0;
		while (layerIndex < nodeCount)
		{

			float a = n->m_OutputBuffer[layerIndex];
			float y = n->m_TargetBuffer[layerIndex];


			float cost = Math::CrossEntropy(a, y);


			n->m_CostBuffer += cost;



			layerIndex++;
		}
	}

	void CrossEntropyDerivativeTest(NetworkPrototype2* n)
	{


		unsigned layerIndex = 0;
		while (layerIndex < n->m_OutputBufferCount)
		{

			float z = n->m_LayerLayout[n->m_LayerLayoutCount-1].Z[layerIndex];
			float a = n->m_OutputBuffer[layerIndex];
			float y = n->m_TargetBuffer[layerIndex];

			float dz = Math::CrossEntropyCostDerivative(z, a, y);

			n->m_LayerLayout[n->m_LayerLayoutCount - 1].dZ[layerIndex] = dz;


			layerIndex++;
		}
	}
	

}