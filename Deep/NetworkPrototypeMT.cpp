#include "pch.h"
#include "NetworkPrototypeMT.h"


namespace TNNT
{

	NetworkPrototypeMT::NetworkPrototypeMT(LayerLayout* layerLayout, FunctionsLayoutMT& functions, unsigned layoutCount, unsigned slaveThreadCount , bool randomizeWeightsAndBiases)
		: m_LayerLayoutCount(layoutCount), m_SlaveThreadCount(slaveThreadCount)
	{


		// 1 layer no network makes; need at least 2
		assert(m_LayerLayoutCount >= 2);

		// A layer of zero nodes would mean you have two separate networks (or a network with 1 less layer, if the input/output layer is missing), 
		// and a layer with a negative numbers of nodes is something I don't want to think about.

		//These are gonna get reused a lot in the constructor.
		unsigned layoutIndex = 0;

	//Getting the LayerLayout ready START

		m_LayerLayout = new LayerLayout[m_LayerLayoutCount];

		layoutIndex = 0;
		while (layoutIndex < m_LayerLayoutCount)
		{
			m_LayerLayout[layoutIndex] = layerLayout[layoutIndex];

			layoutIndex++;
		}



	//Getting the LayerLayout ready STOP




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
			assert(m_LayerLayout[layoutIndex].NodesCount > 0);

			nodesTotal += m_LayerLayout[layoutIndex].NodesCount;
			zTotal += m_LayerLayout[layoutIndex].NodesCount;
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

		m_CostBuffer = new float[m_SlaveThreadCount];
		m_GuessBuffer = new unsigned[m_SlaveThreadCount];

		//NETWORK STRUCTURE (Function layout)

		m_Functions.NeuronFunctions = new FunctionsLayoutMT::NeuronFunction[m_LayerLayoutCount - 1];
		m_Functions.NeuronFunctionsDerivatives = new FunctionsLayoutMT::NeuronFunction[m_LayerLayoutCount - 1];

		m_Functions.FeedForwardCallBackFunctions = new FunctionsLayoutMT::NetworkRelayFunctionMT[m_LayerLayoutCount - 1];
		m_Functions.BackPropegateCallBackFunctionsZ = new FunctionsLayoutMT::NetworkRelayFunctionMT[m_LayerLayoutCount - 1];
		m_Functions.BackPropegateCallBackFunctionsBW = new FunctionsLayoutMT::NetworkRelayFunctionMT[m_LayerLayoutCount - 1];

		m_Functions.CostFunction = functions.CostFunction;
		m_Functions.CostFunctionDerivative = functions.CostFunctionDerivative;
		
		m_Functions.RegularizationFunctions = new FunctionsLayoutMT::NetworkRelayFunctionMT[m_LayerLayoutCount - 1];
		m_Functions.TrainingFunctions = new FunctionsLayoutMT::NetworkRelayFunctionMT[m_LayerLayoutCount - 1];


		//MULTITHREAD MANAGEMENT
		m_SlaveThreads = new std::thread[m_SlaveThreadCount];
		m_Locks = new bool[m_SlaveThreadCount * 2];
		m_SlaveFlags = new bool[m_SlaveThreadCount];


		//Workload Layout (Node pointer is used as the storage pointer)
		m_WorkloadLayout.Nodes = new unsigned[ (2 * m_SlaveThreadCount * m_LayerLayoutCount) + 3*(2 * m_SlaveThreadCount * (m_LayerLayoutCount - 1)) + (2+6)*(2 * m_SlaveThreadCount) ];
		m_WorkloadLayout.Weights = m_WorkloadLayout.Nodes + (2 * m_SlaveThreadCount * (m_LayerLayoutCount));
		m_WorkloadLayout.Biases = m_WorkloadLayout.Weights + (2 * m_SlaveThreadCount * (m_LayerLayoutCount - 1));
		m_WorkloadLayout.Z = m_WorkloadLayout.Biases + (2 * m_SlaveThreadCount * (m_LayerLayoutCount - 1));
		m_WorkloadLayout.Input = m_WorkloadLayout.Z + (2 * m_SlaveThreadCount * (m_LayerLayoutCount - 1));
		m_WorkloadLayout.Output = m_WorkloadLayout.Input + (2 * m_SlaveThreadCount);
		m_WorkloadLayout.WholeNodes = m_WorkloadLayout.Output + (2 * m_SlaveThreadCount);
		m_WorkloadLayout.WholeZ = m_WorkloadLayout.WholeNodes + (2 * m_SlaveThreadCount);
		m_WorkloadLayout.WholeWeights = m_WorkloadLayout.WholeZ + (2 * m_SlaveThreadCount);
		m_WorkloadLayout.WholeBiases = m_WorkloadLayout.WholeWeights + (2 * m_SlaveThreadCount);
		m_WorkloadLayout.WholeInput = m_WorkloadLayout.WholeBiases + (2 * m_SlaveThreadCount);
		m_WorkloadLayout.WholeOutput = m_WorkloadLayout.WholeInput + (2 * m_SlaveThreadCount);


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


		//WORKLOAD LAYOUT SETUP SECTION START

		{


			layoutIndex = 0;
			while (layoutIndex < m_LayerLayoutCount)
			{



				unsigned NStart, NStop, WStart, WStop, BStart, BStop , ZStart, ZStop , OStart, OStop, 
					wholeNStart, wholeNStop, wholeZStart, wholeZStop, wholeWStart, wholeWStop, 
					wholeBStart, wholeBStop, wholeIStart, wholeIStop, wholeOStart, wholeOStop;

				unsigned threadIndex = 0;
				while (threadIndex < m_SlaveThreadCount)
				{
					ThreadWorkloadDivider(NStart, NStop, m_LayerLayout[layoutIndex].NodesCount, threadIndex);

					m_WorkloadLayout.Nodes[(layoutIndex) * (2 * m_SlaveThreadCount) + 2 * threadIndex] = NStart ;
					m_WorkloadLayout.Nodes[(layoutIndex) * (2 * m_SlaveThreadCount) + 2 * threadIndex + 1] = NStop ;

					if (layoutIndex == 0)
					{
						m_WorkloadLayout.Input[2 * threadIndex] = NStart;
						m_WorkloadLayout.Input[2 * threadIndex + 1] = NStop;

					}


					if (layoutIndex != 0)
					{
						ThreadWorkloadDivider(WStart, WStop, m_LayerLayout[layoutIndex].WeightsCount, threadIndex);
						ThreadWorkloadDivider(BStart, BStop, m_LayerLayout[layoutIndex].BiasesCount, threadIndex);
						ThreadWorkloadDivider(ZStart, ZStop, m_LayerLayout[layoutIndex].NodesCount, threadIndex);

						m_WorkloadLayout.Weights[(layoutIndex - 1) * (2 * m_SlaveThreadCount) + 2 * threadIndex] = WStart ;
						m_WorkloadLayout.Weights[(layoutIndex - 1) * (2 * m_SlaveThreadCount) + 2 * threadIndex + 1] = WStop ;

						m_WorkloadLayout.Biases[(layoutIndex - 1) * (2 * m_SlaveThreadCount) + 2 * threadIndex] = BStart ;
						m_WorkloadLayout.Biases[(layoutIndex - 1) * (2 * m_SlaveThreadCount) + 2 * threadIndex + 1] = BStop ;


						m_WorkloadLayout.Z[(layoutIndex-1) * (2 * m_SlaveThreadCount) + 2 * threadIndex] = ZStart ;
						m_WorkloadLayout.Z[(layoutIndex-1) * (2 * m_SlaveThreadCount) + 2 * threadIndex + 1] = ZStop ;




					}

					if (layoutIndex == m_LayerLayoutCount - 1)
					{
						ThreadWorkloadDivider(OStart, OStop, m_LayerLayout[layoutIndex].NodesCount, threadIndex);

						m_WorkloadLayout.Output[2 * threadIndex] = OStart;
						m_WorkloadLayout.Output[2 * threadIndex + 1] = OStop;

						// Handle the whole arrays here, since they dont need to be calculated layer by layer.
						ThreadWorkloadDivider(wholeNStart, wholeNStop, m_ACount, threadIndex);
						ThreadWorkloadDivider(wholeZStart, wholeZStop, m_ZCount, threadIndex);
						ThreadWorkloadDivider(wholeWStart, wholeWStop, m_WeightsCount, threadIndex);
						ThreadWorkloadDivider(wholeBStart, wholeBStop, m_BiasesCount, threadIndex);
						ThreadWorkloadDivider(wholeIStart, wholeIStop, m_InputBufferCount, threadIndex);
						ThreadWorkloadDivider(wholeOStart, wholeOStop, m_OutputBufferCount, threadIndex);

						m_WorkloadLayout.WholeNodes[2 * threadIndex] = wholeNStart;
						m_WorkloadLayout.WholeNodes[2 * threadIndex + 1] = wholeNStop;
							
						m_WorkloadLayout.WholeZ[2 * threadIndex] = wholeZStart;
						m_WorkloadLayout.WholeZ[2 * threadIndex + 1] = wholeZStop;

						m_WorkloadLayout.WholeWeights[2 * threadIndex] = wholeWStart;
						m_WorkloadLayout.WholeWeights[2 * threadIndex + 1] = wholeWStop;
						
						m_WorkloadLayout.WholeBiases[2 * threadIndex] = wholeBStart;
						m_WorkloadLayout.WholeBiases[2 * threadIndex + 1] = wholeBStop;

						m_WorkloadLayout.WholeInput[2 * threadIndex] = wholeIStart;
						m_WorkloadLayout.WholeInput[2 * threadIndex + 1] = wholeIStop;

						m_WorkloadLayout.WholeOutput[2 * threadIndex] = wholeOStart;
						m_WorkloadLayout.WholeOutput[2 * threadIndex + 1] = wholeOStop;
					}


					threadIndex++;
				}




				layoutIndex++;
			}


		}

		//WORKLOAD LAYOUT SETUP SECTION STOP


		//SPINLOCK SETUP START
		unsigned flagIndex = 0;
		while (flagIndex < m_SlaveThreadCount)
		{
			m_Locks[flagIndex * 2] = false;
			m_Locks[flagIndex * 2 + 1] = false;
			m_SlaveFlags[flagIndex] = false;
			flagIndex++;
		}
		//SPINLOCK SETUP STOP


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
				m_Biases[index ] = temp;

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

		{
			unsigned thread = 0;
			while (thread < m_SlaveThreadCount)
			{
				m_SlaveThreads[thread] = std::thread(&NetworkPrototypeMT::SetTempToBiasesAndWeights, this, thread);
				thread++;
			}

			thread = 0;
			while (thread < m_SlaveThreadCount)
			{
				m_SlaveThreads[thread].join();
				thread++;
			}
		}
		//WEIGHTS AND BIASES SETUP STOP



	//ENSURING THAT CERTAIN INTEGER AND FLOAT ARRAYS HAVE ACCEPTABLE INITIAL VALUES STOP

	}

	NetworkPrototypeMT::~NetworkPrototypeMT()
	{

		delete[] m_LayerLayout;

		delete[] m_NetworkFixedData;

		delete[] m_CostBuffer;
		delete[] m_GuessBuffer;


		delete[] m_SlaveThreads;
		delete[] m_Locks;
		delete[] m_SlaveFlags;

	}

	float NetworkPrototypeMT::CheckSuccessRate()
	{
		return CheckSuccessRateMasterFunction();
	}

	float NetworkPrototypeMT::CheckCost()
	{
		return CheckCostMasterFunction();
	}

	void NetworkPrototypeMT::Train(DataSet* data, HyperParameters& params)
	{
		SetData(data);
		SetHyperParameters(params);
		TrainMasterFunction();

	}

	unsigned NetworkPrototypeMT::Check(float* input)
	{
		return CheckMasterFunction(input);
	}








	void NetworkPrototypeMT::SetTempToWeights(unsigned thread)
	{

		unsigned start = m_WorkloadLayout.WholeWeights[2 * thread];
		unsigned stop = m_WorkloadLayout.WholeWeights[2 * thread + 1];



		unsigned dist = stop - start;

		memcpy(&m_TempWeights[start], &m_Weights[start], dist * sizeof(float));

	}

	void NetworkPrototypeMT::SetWeightsToTemp(unsigned thread)
	{

		unsigned start = m_WorkloadLayout.WholeWeights[2 * thread];
		unsigned stop = m_WorkloadLayout.WholeWeights[2 * thread + 1];



		unsigned dist = stop - start;

		memcpy(&m_Weights[start], &m_TempWeights[start], dist * sizeof(float));


	}


	void NetworkPrototypeMT::SetTempToBiases(unsigned thread)
	{


		unsigned start = m_WorkloadLayout.WholeBiases[2 * thread];
		unsigned stop = m_WorkloadLayout.WholeBiases[2 * thread + 1];



		unsigned dist = stop - start;

		memcpy(&m_TempBiases[start], &m_Biases[start], dist * sizeof(float));


	}

	void NetworkPrototypeMT::SetBiasesToTemp(unsigned thread)
	{

		unsigned start = m_WorkloadLayout.WholeBiases[2 * thread];
		unsigned stop = m_WorkloadLayout.WholeBiases[2 * thread + 1];



		unsigned dist = stop - start;

		memcpy(&m_Biases[start], &m_TempBiases[start], dist * sizeof(float));
	}


	void NetworkPrototypeMT::SetTempToBiasesAndWeights(unsigned thread)
	{
		SetTempToBiases(thread);
		SetTempToWeights(thread);
	}





	void NetworkPrototypeMT::SetData(DataSet* data)
	{
		delete[] m_Indices;

		m_Data = data;

		m_Indices = new unsigned[m_Data->TrainingCount ];


		unsigned index = 0;
		while (index < m_Data->TrainingCount)
		{

			m_Indices[index] = index;

			index++;
		}

	}

	//TODO figure out whether you want to keep this or not.
	void NetworkPrototypeMT::ResetIndices(unsigned thread)
	{
		unsigned start;
		unsigned stop;
		ThreadWorkloadDivider(start, stop, m_Data->TrainingCount, thread);

		

		unsigned index = start;
		while (index < stop)
		{
			
			m_Indices[index] = index;
			
			index++;
		}
	}

	void NetworkPrototypeMT::SetHyperParameters(HyperParameters& params)
	{


		m_HyperParameters = params;

	}

	void NetworkPrototypeMT::SetInput(float* input, unsigned thread)
	{
		unsigned start = m_WorkloadLayout.WholeInput[2 * thread ];
		unsigned stop = m_WorkloadLayout.WholeInput[2 * thread + 1];

		unsigned dist = stop - start;
		

		memcpy(&m_InputBuffer[start], &input[start], dist * sizeof(float));


	}

	void NetworkPrototypeMT::SetTarget(float* target, unsigned thread)
	{
		unsigned start = m_WorkloadLayout.WholeOutput[2 * thread];
		unsigned stop = m_WorkloadLayout.WholeOutput[2 * thread + 1];

		unsigned dist = stop - start;


		memcpy(&m_TargetBuffer[start], &target[start], dist * sizeof(float));
			

	}

	void NetworkPrototypeMT::ThreadWorkloadDivider(unsigned& start, unsigned& stop, unsigned workCount, unsigned thread)
	{
		start = 0;
		stop = 0;

		unsigned workloadCount = (workCount / m_SlaveThreadCount);
		unsigned workloadRemainder = (workCount % m_SlaveThreadCount);


		if (thread < workloadRemainder)
		{
			start = (workloadCount + 1) * thread; // add padding * thread to this, if you want padding
			stop = start + (workloadCount + 1);
		}
		else
		{
			start = workloadCount * thread + workloadRemainder; // add padding * thread to this, if you want padding

			stop = start + workloadCount;
		}
	}



	void NetworkPrototypeMT::SpinLock(unsigned thread)
	{

		unsigned i = 0;
		while (i < m_SlaveThreadCount)
		{
			if (i == thread)
			{
				m_Locks[i + m_SlaveThreadCount] = false;
				m_Locks[i] = true;
			}
			if (m_Locks[i])
			{
				i++;
			}
			
		}
		while (i < 2 * m_SlaveThreadCount)
		{
			if (i == (thread + m_SlaveThreadCount))
			{
				m_Locks[i - m_SlaveThreadCount] = false;
				m_Locks[i] = true;
			}
			if (m_Locks[i])
			{
				i++;
			}
			
		}
	}

	void NetworkPrototypeMT::SlaveControlStation(unsigned position)
	{

		while (position > m_MasterControlPoint)
		{
			;//About as empty as my head.
		}
	}

	void NetworkPrototypeMT::WaitForSlaves()
	{
		unsigned i = 0;
		while (i < m_SlaveThreadCount)
		{
			if (m_SlaveFlags[i])
			{
				m_SlaveFlags[i] = false;
				i++;
			}
		}

	}

	void NetworkPrototypeMT::FeedForward(unsigned thread)
	{

		if (thread == 0)
		{
			m_LayerLayoutPosition = 1;

		}
		SpinLock(thread);
		unsigned layoutIndex = 1;

		while (layoutIndex < m_LayerLayoutCount)
		{

			//No function for the inputlayer, which means that the function corresponding to any other layer is located at layoutIndex - 1.
			m_Functions.FeedForwardCallBackFunctions[layoutIndex - 1].f(this, thread);
			SpinLock(thread);

			//PositionData
			if (thread == 0)
			{

				m_LayerLayoutPosition++;

			}
			
			SpinLock(thread);
			layoutIndex++;
			
		}



	}

	void NetworkPrototypeMT::Backpropegate(unsigned thread)
	{

		unsigned lastLayer = m_LayerLayoutCount - 1;

		if (thread == 0)
		{
			m_LayerLayoutPosition = lastLayer;

		}
		SpinLock(thread);

		

		unsigned reveresLayoutIndex = 0;
		while (reveresLayoutIndex < lastLayer)
		{

			m_Functions.BackPropegateCallBackFunctionsZ[(lastLayer-1) - reveresLayoutIndex].f(this, thread);
			SpinLock(thread);

			m_Functions.BackPropegateCallBackFunctionsBW[(lastLayer - 1) - reveresLayoutIndex].f(this, thread);
			SpinLock(thread);

			if(thread == 0)
			{
				m_LayerLayoutPosition--;

			}
			SpinLock(thread);




			
			reveresLayoutIndex++;
		}

	}

	void NetworkPrototypeMT::Regularization(unsigned thread)
	{
		if (thread == 0)
		{
			m_LayerLayoutPosition = 1;

		}
		SpinLock(thread);
		unsigned layoutIndex = 1;

		while (layoutIndex < m_LayerLayoutCount)
		{

			//No function for the inputlayer, which means that the function corresponding to any other layer is located at layoutIndex - 1.
			m_Functions.RegularizationFunctions[layoutIndex - 1].f(this, thread);
			SpinLock(thread);

			//PositionData
			if (thread == 0)
			{

				m_LayerLayoutPosition++;

			}

			SpinLock(thread);
			layoutIndex++;

		}
	}

	void NetworkPrototypeMT::Train(unsigned thread)
	{
		if (thread == 0)
		{
			m_LayerLayoutPosition = 1;

		}
		SpinLock(thread);
		unsigned layoutIndex = 1;

		while (layoutIndex < m_LayerLayoutCount)
		{

			//No function for the inputlayer, which means that the function corresponding to any other layer is located at layoutIndex - 1.
			m_Functions.TrainingFunctions[layoutIndex - 1].f(this, thread);
			SpinLock(thread);

			//PositionData
			if (thread == 0)
			{

				m_LayerLayoutPosition++;

			}

			SpinLock(thread);
			layoutIndex++;

		}
	}


	void NetworkPrototypeMT::TrainOnSet(unsigned batchCount, unsigned batch, unsigned thread)
	{
		SpinLock(thread);
		Regularization(thread);
		
		
		

		unsigned exampleIndex = 0;
		while (exampleIndex < batchCount)
		{


			
			unsigned index = m_Indices[exampleIndex + batch * m_HyperParameters.BatchCount];
			SetInput(&(m_Data->TrainingInputs[index * m_InputBufferCount]), thread);
			SetTarget(&(m_Data->TraningTargets[index * m_OutputBufferCount]), thread);
			
			SpinLock(thread);

			FeedForward(thread);
			Backpropegate(thread);

			Train(thread);



			exampleIndex++;


		}

		SpinLock(thread);
		SetBiasesToTemp(thread);
		SetWeightsToTemp(thread);
	
	}

	void NetworkPrototypeMT::TrainSlaveFunction( unsigned thread)
	{

		const unsigned batchTotal = m_Data->TrainingCount / m_HyperParameters.BatchCount;
		const unsigned remainingBatchCount = m_Data->TrainingCount % m_HyperParameters.BatchCount;

		unsigned position = 0;

		unsigned epoch = 0;
		while (epoch < m_HyperParameters.Epochs)
		{

			unsigned batch = 0;
			while (batch < batchTotal)
			{
				position++;

				SlaveControlStation(position);
				TrainOnSet(m_HyperParameters.BatchCount, batch, thread);

				batch++;
			}


			if(remainingBatchCount !=0)
			{
				

				position++;
				SlaveControlStation(position);

				TrainOnSet(remainingBatchCount, batch , thread);
			}

			position = 0;
			m_SlaveFlags[thread] = true;
			epoch++;
		}

	}

	void NetworkPrototypeMT::TrainMasterFunction()
	{
		//Timer start
		auto start = std::chrono::high_resolution_clock::now();

		m_MasterControlPoint = 0;

		unsigned thread = 0;
		while (thread < m_SlaveThreadCount)
		{
			m_SlaveThreads[thread] = std::thread(&NetworkPrototypeMT::TrainSlaveFunction, this, thread);
			thread++;
		}



		const unsigned batchNum = m_Data->TrainingCount / m_HyperParameters.BatchCount;
		const unsigned remainingBatch = m_Data->TrainingCount % m_HyperParameters.BatchCount;

		std::mt19937 mt;

		unsigned epoch = 0;
		while (epoch < m_HyperParameters.Epochs)
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
				
				if (batch >= 64)
				{
					m_MasterControlPoint++;
				}
				
				
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

				if (batch >= 64)
				{
					m_MasterControlPoint++;
				}
				batch++;
				
			}

			m_MasterControlPoint += 64;

			
			WaitForSlaves();
			m_MasterControlPoint = 0;

			epoch++;
		}

		thread = 0;
		while (thread < m_SlaveThreadCount)
		{
			m_SlaveThreads[thread].join();
			thread++;
		}

		//Timer stop
		auto stop = std::chrono::high_resolution_clock::now();
		std::chrono::duration<float>  time = stop - start;
		m_LastTime[0] = time.count();
	}

	void NetworkPrototypeMT::CheckCostSlaveFunction(unsigned thread)
	{
		unsigned checkIndex = 0;
		while (checkIndex < m_Data->TestCount)
		{

			SetInput(&m_Data->TestInputs[checkIndex * m_InputBufferCount], thread);
			SetTarget(&m_Data->TestTargets[checkIndex * m_OutputBufferCount], thread);
			SpinLock(thread);

			FeedForward(thread);

			m_Functions.CostFunction.f(this, thread);
			SpinLock(thread);
			
			checkIndex++;
		}

		
	}

	float NetworkPrototypeMT::CheckCostMasterFunction()
	{
		auto start = std::chrono::high_resolution_clock::now();

		float cost = 0;
		

		unsigned thread = 0;
		while (thread < m_SlaveThreadCount)
		{
			m_CostBuffer[thread] = 0;
			m_SlaveThreads[thread] = std::thread(&NetworkPrototypeMT::CheckCostSlaveFunction, this, thread);
			thread++;
		}

		thread = 0;
		while (thread < m_SlaveThreadCount)
		{
			m_SlaveThreads[thread].join();
			cost += m_CostBuffer[thread];

			thread++;
		}


		auto stop = std::chrono::high_resolution_clock::now();
		std::chrono::duration<float>  time = stop - start;
		m_LastTime[1] = time.count();


		auto returnCost = cost / ((float)m_Data->TestCount);

		return returnCost;

	}

	void NetworkPrototypeMT::CheckSuccessRateSlaveFunction(unsigned thread)
	{
		

		

		unsigned checkIndex = 0;
		while (checkIndex < m_Data->TestCount)
		{
			
			SetInput(&m_Data->TestInputs[checkIndex * m_InputBufferCount], thread);
			SpinLock(thread);
			
			FeedForward(thread);
			
			
			

			int championItterator = -1;
			float champion = 0;

			unsigned start = m_WorkloadLayout.WholeOutput[2 * thread];
			unsigned stop = m_WorkloadLayout.WholeOutput[2 * thread + 1];

			unsigned outputIndex = start;
			while (outputIndex < stop)
			{

				if (m_OutputBuffer[outputIndex] >= champion)
				{
					champion = m_OutputBuffer[outputIndex];
					championItterator = outputIndex;
				}


				outputIndex++;
			}
			if (championItterator == -1)
			{
				
				assert(false);
				
			}
			m_GuessBuffer[thread] = championItterator;

			
			m_SlaveFlags[thread] = true;
			

			
			checkIndex++;
			SlaveControlStation(checkIndex);
		}

		
	}

	float NetworkPrototypeMT::CheckSuccessRateMasterFunction()
	{
		auto start = std::chrono::high_resolution_clock::now();


		m_MasterControlPoint = 0;
		unsigned thread = 0;
		while (thread < m_SlaveThreadCount)
		{
			m_SlaveThreads[thread] = std::thread(&NetworkPrototypeMT::CheckSuccessRateSlaveFunction, this, thread);
			thread++;
		}

		
		float score = 0.0f;

		unsigned checkIndex = 0;
		while (checkIndex < m_Data->TestCount)
		{
			int championItterator = -1;
			float champion = 0;


			WaitForSlaves();

			unsigned lim = (m_SlaveThreadCount < m_OutputBufferCount) ? m_SlaveThreadCount : m_OutputBufferCount;

			unsigned threadItt = 0;
			while (threadItt < lim)
			{
				unsigned itt = m_GuessBuffer[threadItt];
				if (m_OutputBuffer[itt] >= champion)
				{
					

					champion = m_OutputBuffer[itt];
					championItterator = itt;
				}


				threadItt++;
			}

			if (m_Data->TestTargets[m_OutputBufferCount * checkIndex + championItterator] == 1)
			{
				score += 1.0f;
			}

			m_MasterControlPoint++;

			checkIndex++;
		}

		float rate = score / ((float)m_Data->TestCount);

		

		thread = 0;
		while (thread < m_SlaveThreadCount)
		{
			m_SlaveThreads[thread].join();
			thread++;
		}

		auto stop = std::chrono::high_resolution_clock::now();
		std::chrono::duration<float>  time = stop - start;
		m_LastTime[2] = time.count();

		return rate;

	}

	void NetworkPrototypeMT::CheckSlaveFunction(float* input, unsigned thread)
	{


		SetInput(input, thread);
		SpinLock(thread);
		FeedForward(thread);

		int championItterator = -1;
		float champion = 0;



		unsigned start = m_WorkloadLayout.WholeOutput[2 * thread];
		unsigned stop = m_WorkloadLayout.WholeOutput[2 * thread + 1 ];

	

		unsigned outputIndex = start;
		while (outputIndex < stop)
		{

			if (m_OutputBuffer[outputIndex] >= champion)
			{
				champion = m_OutputBuffer[outputIndex];
				championItterator = outputIndex;
			}


			outputIndex++;
		}
		if (championItterator == -1)
		{
			assert(false);
		}

		m_GuessBuffer[thread] = championItterator;


		m_SlaveFlags[thread] = true;

		
		
		
	}

	unsigned NetworkPrototypeMT::CheckMasterFunction(float* input)
	{
		auto start = std::chrono::high_resolution_clock::now();


		

		unsigned thread = 0;
		while (thread < m_SlaveThreadCount)
		{
			m_SlaveThreads[thread] = std::thread(&NetworkPrototypeMT::CheckSlaveFunction, this, input, thread);
			thread++;
		}



		int championItterator = -1;
		float champion = 0;

		WaitForSlaves();

		unsigned lim = (m_SlaveThreadCount < m_OutputBufferCount) ? m_SlaveThreadCount : m_OutputBufferCount;

		unsigned threadItt = 0;
		while (threadItt < lim)
		{
			unsigned itt = m_GuessBuffer[threadItt];
			if (m_OutputBuffer[itt] >= champion)
			{
				champion = m_OutputBuffer[itt];
				championItterator = itt;
			}

			threadItt++;
		}

		thread = 0;
		while (thread < m_SlaveThreadCount)
		{
			m_SlaveThreads[thread].join();
			thread++;
		}

		auto stop = std::chrono::high_resolution_clock::now();
		std::chrono::duration<float>  time = stop - start;
		m_LastTime[2] = time.count();

		return championItterator;
	}


}