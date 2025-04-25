#include "pch.h"
#include "NetworkPrototype.h"
#include "LayerFunctions.h"

namespace TNNT
{
	//Constructors And destructor

	NetworkPrototype::NetworkPrototype(LayerLayout* layerLayout, FunctionsLayout& functions, unsigned layoutCount, bool randomizeWeightsAndBiases)
		: m_LayerLayoutCount(layoutCount)
	{


		// 1 layer no network makes; need at least 2
		assert(m_LayerLayoutCount >= 2);

		// A layer of zero nodes would mean you have two separate networks (or a network with 1 less layer, if the input/output layer is missing), 
		// and a layer with a negative numbers of nodes is something I don't want to think about.

		//These are gonna get reused a lot in the constructor.
		unsigned layoutIndex = 0;



		// DETERMENING THE COUNT OF MOST ARRAYS AND CALCULATING OTHER IMPORTANT INTEGERS START



		//Keep in mind that there aren't supposed to be any biases or weights in the 0th layer, so their count for that layer should both be 0.
		unsigned nodesTotal = 0;
		unsigned zTotal = 0;
		unsigned biasTotal = 0;
		unsigned weightTotal = 0;


		layoutIndex = 0;
		while (layoutIndex < m_LayerLayoutCount)
		{
			// A layer of zero nodes would mean you have two separate networks (or a network with 1 less layer, if the input/output layer is missing), 
			// and a layer with a negative numbers of nodes is something I don't want to think about.
			assert(layerLayout[layoutIndex].NodesCount > 0);

			nodesTotal += layerLayout[layoutIndex].NodesCount;
			zTotal += layerLayout[layoutIndex].ZCount;
			biasTotal += layerLayout[layoutIndex].BiasesCount;
			weightTotal += layerLayout[layoutIndex].WeightsCount;

			layoutIndex++;
		}

		m_ACount = nodesTotal;
		m_ZCount = zTotal;
		m_BiasesCount = biasTotal;
		m_WeightsCount = weightTotal;

		m_InputBufferCount = layerLayout[0].NodesCount;
		m_OutputBufferCount = layerLayout[m_LayerLayoutCount - 1].NodesCount;



		// DETERMENING THE COUNT OF MOST ARRAYS AND CALCULATING OTHER IMPORTANT INTEGERS STOP

		constexpr size_t alignment = 64 / 8;

		//MEMORY ALLOCATION AND POINTER SETUP START

		//Order: A, Weights, Biases, Z, dZ, WeightsBuffer, BiasesBuffer, dWeights, dBiases,  Target

		unsigned layerLayoutChunkSize = m_LayerLayoutCount * sizeof(LayerLayout) + (alignment - (m_LayerLayoutCount * sizeof(LayerLayout)) % alignment);

		m_NetworkFixedDataSize = 1 * (m_ACount) * sizeof(float)
			+ 4 * (m_WeightsCount) * sizeof(float) + 3 * (m_BiasesCount) * sizeof(float)
			+ 2 * (m_ZCount) * sizeof(float) + 1 * (m_OutputBufferCount) * sizeof(float)
			+ 1 * layerLayoutChunkSize;

		m_NetworkFixedData = (char*)allocate_aligned(m_NetworkFixedDataSize, alignment);




		//NETWORK STRUCTURE
		m_A = (float*)(m_NetworkFixedData);
		m_InputBuffer = m_A;
		m_OutputBuffer = m_A + m_ACount - m_OutputBufferCount;

		m_Weights = m_A + m_ACount;
		m_Biases = m_Weights + m_WeightsCount;

		m_Z = m_Biases + m_BiasesCount;
		m_DeltaZ = m_Z + m_ZCount;

		m_WeightsTranspose = m_DeltaZ + m_ZCount;

		m_TempWeights = m_WeightsTranspose + m_WeightsCount;
		m_TempBiases = m_TempWeights + m_WeightsCount;

		m_DeltaWeights = m_TempBiases + m_BiasesCount;
		m_DeltaBiases = m_DeltaWeights + m_WeightsCount;

		//EVALUATION BUFFERS
		m_TargetBuffer = m_DeltaBiases + m_BiasesCount;



		//Getting the LayerLayout ready START

		unsigned padding = alignof(LayerLayout) - (((size_t)m_TargetBuffer) % alignof(LayerLayout));
		pr(padding);

		m_LayerLayout = (LayerLayout*)(((char*)(m_TargetBuffer + m_OutputBufferCount)) + padding);

		pr(m_TargetBuffer);
		pr(padding);
		pr(m_LayerLayout);

		layoutIndex = 0;
		while (layoutIndex < m_LayerLayoutCount)
		{
			m_LayerLayout[layoutIndex] = layerLayout[layoutIndex];

			layoutIndex++;
		}
		//Getting the LayerLayout ready STOP



		//NETWORK STRUCTURE (Function layout)

		m_Functions.NeuronFunctions = new FunctionsLayout::NeuronFunction[m_LayerLayoutCount - 1];
		m_Functions.NeuronFunctionsDerivatives = new FunctionsLayout::NeuronFunction[m_LayerLayoutCount - 1];

		m_Functions.FeedForwardCallBackFunctions = new FunctionsLayout::NetworkRelayFunction[m_LayerLayoutCount - 1];
		m_Functions.BackPropegateCallBackFunctionsZ = new FunctionsLayout::NetworkRelayFunction[m_LayerLayoutCount - 1];
		m_Functions.BackPropegateCallBackFunctionsBW = new FunctionsLayout::NetworkRelayFunction[m_LayerLayoutCount - 1];

		m_Functions.CostFunction = functions.CostFunction;
		m_Functions.CostFunctionDerivative = functions.CostFunctionDerivative;

		m_Functions.RegularizationFunctions = new FunctionsLayout::NetworkRelayFunction[m_LayerLayoutCount - 1];
		m_Functions.TrainingFunctions = new FunctionsLayout::NetworkRelayFunction[m_LayerLayoutCount - 1];




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
				m_LayerLayout[layoutIndex].WeightsTranspose = m_WeightsTranspose + aAdjustWeights;

				aAdjustWeights += m_LayerLayout[layoutIndex].WeightsCount;


				m_LayerLayout[layoutIndex].Biases = m_Biases + aAdjustBiases;
				m_LayerLayout[layoutIndex].dBiases = m_DeltaBiases + aAdjustBiases;
				m_LayerLayout[layoutIndex].TempBiases = m_TempBiases + aAdjustBiases;

				aAdjustBiases += m_LayerLayout[layoutIndex].BiasesCount;



				layoutIndex++;
			}
		}


		//MEMORY ALLOCATION AND POINTER SETUP STOP



		//FUNCTION LAYOUT SETUP START

		layoutIndex = 0;
		while (layoutIndex < m_LayerLayoutCount)
		{

			if (layoutIndex < m_LayerLayoutCount - 1)
			{
				m_Functions.NeuronFunctions[layoutIndex] = functions.NeuronFunctions[layoutIndex];
				m_Functions.NeuronFunctionsDerivatives[layoutIndex] = functions.NeuronFunctionsDerivatives[layoutIndex];

				m_Functions.FeedForwardCallBackFunctions[layoutIndex] = functions.FeedForwardCallBackFunctions[layoutIndex];

				m_Functions.BackPropegateCallBackFunctionsBW[layoutIndex] = functions.BackPropegateCallBackFunctionsBW[layoutIndex];
				m_Functions.BackPropegateCallBackFunctionsZ[layoutIndex] = functions.BackPropegateCallBackFunctionsZ[layoutIndex];

				m_Functions.RegularizationFunctions[layoutIndex] = functions.RegularizationFunctions[layoutIndex];
				m_Functions.TrainingFunctions[layoutIndex] = functions.TrainingFunctions[layoutIndex];

			}

			layoutIndex++;
		}

		//FUNCTION LAYOUT STOP



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
		ResetTranspose();
		//WEIGHTS AND BIASES SETUP STOP



	//ENSURING THAT CERTAIN INTEGER AND FLOAT ARRAYS HAVE ACCEPTABLE INITIAL VALUES STOP

	}

	NetworkPrototype::~NetworkPrototype()
	{

		_aligned_free(m_NetworkFixedData);


		delete[] m_Indices;
	}




	float NetworkPrototype::CheckSuccessRate()
	{
		return CheckSuccessRateMasterFunction();
	}

	float NetworkPrototype::CheckCost()
	{
		return CheckCostMasterFunction();
	}

#pragma warning(disable : 4996)
	void NetworkPrototype::SaveParams()
	{
		FILE* paramFile = std::fopen("params.bin", "rb");


		std::fread((char*)m_Biases, sizeof(float), m_BiasesCount, paramFile);
		std::fread((char*)m_Weights, sizeof(float), m_WeightsCount, paramFile);

		fclose(paramFile);

	}

	void NetworkPrototype::LoadParams()
	{
		FILE* paramFile = std::fopen("params.bin", "wb");

		std::fwrite((char*)m_Biases, sizeof(float), m_BiasesCount, paramFile);
		std::fwrite((char*)m_Weights, sizeof(float), m_WeightsCount, paramFile);

		fclose(paramFile);

		SetTempToWeights();
		SetTempToBiases();



	}

	void NetworkPrototype::Train(DataSet* data, HyperParameters& params)
	{
		SetData(data);
		SetHyperParameters(params);
		TrainMasterFunction();
	}

	unsigned NetworkPrototype::Check(float* input)
	{
		return(CheckMasterFunction(input));
	}





	//Public functions


	// Private functions

	void NetworkPrototype::SetBiasesToTemp()
	{

		memcpy(m_Biases, m_TempBiases, sizeof(float) * m_BiasesCount);
	}

	void NetworkPrototype::SetTempToBiases()
	{

		memcpy(m_TempBiases, m_Biases, sizeof(float) * m_BiasesCount);

	}


	void NetworkPrototype::SetWeightsToTemp()
	{

		memcpy(m_Weights, m_TempWeights, sizeof(float) * m_WeightsCount);
		ResetTranspose();

	}

	void NetworkPrototype::SetTempToWeights()
	{

		memcpy(m_TempWeights, m_Weights, sizeof(float) * m_WeightsCount);
	}

	void NetworkPrototype::ResetTranspose()
	{
		unsigned layerLayoutIndex = 1;
		while (layerLayoutIndex < m_LayerLayoutCount)
		{

			LayerLayout prevLayer = m_LayerLayout[layerLayoutIndex - 1];
			LayerLayout currentLayer = m_LayerLayout[layerLayoutIndex];

			unsigned prevLayerIndex = 0;
			while (prevLayerIndex < prevLayer.NodesCount)
			{
				unsigned currentLayerIndex = 0;
				while (currentLayerIndex < currentLayer.NodesCount)
				{

					currentLayer.WeightsTranspose[currentLayer.NodesCount * prevLayerIndex + currentLayerIndex] = currentLayer.Weights[prevLayer.NodesCount * currentLayerIndex + prevLayerIndex];
					currentLayerIndex++;
				}
				prevLayerIndex++;
			}
			layerLayoutIndex++;
		}


	}




	void NetworkPrototype::SetData(DataSet* data)
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



	void NetworkPrototype::SetHyperParameters(HyperParameters& params)
	{

		m_HyperParameters = params;


	}


	void NetworkPrototype::SetInput(const float* input)
	{

		memcpy(m_InputBuffer, input, sizeof(float) * m_InputBufferCount);

	}

	void NetworkPrototype::SetTarget(const float* target)
	{



		memcpy(m_TargetBuffer, target, sizeof(float) * m_OutputBufferCount);


	}


	void NetworkPrototype::FeedForward()
	{



		m_LayerLayoutPosition = 1;



		unsigned layoutIndex = 1;

		while (layoutIndex < m_LayerLayoutCount)
		{

			//No function for the inputlayer, which means that the function corresponding to any other layer is located at layer - 1.
			m_Functions.FeedForwardCallBackFunctions[layoutIndex - 1].f(this);


			m_LayerLayoutPosition++;

			layoutIndex++;

		}

	}

	void NetworkPrototype::Backpropegate()
	{

		unsigned lastLayer = m_LayerLayoutCount - 1;



		m_LayerLayoutPosition = lastLayer;



		unsigned reveresLayoutIndex = 0;
		while (reveresLayoutIndex < lastLayer)
		{

			m_Functions.BackPropegateCallBackFunctionsZ[(lastLayer - 1) - reveresLayoutIndex].f(this);


			m_Functions.BackPropegateCallBackFunctionsBW[(lastLayer - 1) - reveresLayoutIndex].f(this);

			m_LayerLayoutPosition--;


			reveresLayoutIndex++;
		}

	}

	void NetworkPrototype::Regularization()
	{


		m_LayerLayoutPosition = 1;



		unsigned layoutIndex = 1;

		while (layoutIndex < m_LayerLayoutCount)
		{


			m_Functions.RegularizationFunctions[layoutIndex - 1].f(this);






			m_LayerLayoutPosition++;




			layoutIndex++;

		}
	}

	void NetworkPrototype::Train()
	{

		m_LayerLayoutPosition = 1;



		unsigned layoutIndex = 1;

		while (layoutIndex < m_LayerLayoutCount)
		{


			m_Functions.TrainingFunctions[layoutIndex - 1].f(this);






			m_LayerLayoutPosition++;




			layoutIndex++;

		}
	}



	void NetworkPrototype::TrainOnSet(unsigned batchCount, unsigned batch)
	{


		Regularization();






		unsigned exampleIndex = 0;
		while (exampleIndex < batchCount)
		{

			unsigned index = m_Indices[exampleIndex + batch * m_HyperParameters.BatchCount];
			SetInput(&(m_Data->TrainingInputs[index * m_InputBufferCount]));
			SetTarget(&(m_Data->TraningTargets[index * m_OutputBufferCount]));


			FeedForward();
			Backpropegate();

			Train();





			exampleIndex++;


		}


		SetBiasesToTemp();
		SetWeightsToTemp();

	}


	void NetworkPrototype::TrainMasterFunction()
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

					unsigned randomIndex = (mt() % randomIndexCount) + randomIndexPos;

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


	float NetworkPrototype::CheckCostMasterFunction()
	{

		auto start = std::chrono::high_resolution_clock::now();


		m_CostBuffer = 0;

		unsigned checkIndex = 0;
		while (checkIndex < m_Data->TestCount)
		{

			SetInput(&m_Data->TestInputs[checkIndex * m_InputBufferCount]);
			SetTarget(&m_Data->TestTargets[checkIndex * m_OutputBufferCount]);


			FeedForward();

			m_Functions.CostFunction.f(this);


			checkIndex++;
		}



		auto stop = std::chrono::high_resolution_clock::now();
		std::chrono::duration<float>  time = stop - start;
		m_LastTime[1] = time.count();



		return  m_CostBuffer / ((float)m_Data->TestCount);

	}

	float NetworkPrototype::CheckSuccessRateMasterFunction()
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

	unsigned NetworkPrototype::CheckMasterFunction(float* input)
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


}