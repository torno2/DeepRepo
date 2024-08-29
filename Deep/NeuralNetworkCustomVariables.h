#pragma once
#include "TNNTConstantsAndDefines.h"

#include "Control.h"


namespace TNNT
{


	struct LayerLayout
	{
		float* A;
		float* Z;
		float* Weights;
		float* Biases;
		
		float* dZ;
		float* dWeights;
		float* dBiases;
		float* TempWeights;
		float* TempBiases;

		unsigned* LayerDim;

		unsigned* KerDim;
		unsigned* Stride;
		unsigned* Padding;


		unsigned NodesCount;
		unsigned ZCount;
		unsigned BiasesCount;
		unsigned WeightsCount;



		unsigned SubLayerCount = 1;
		unsigned LayerDimCount = 1;

		unsigned KerDimCount = 1;


		float LearningRate = 0.01f;
		float RegularizationConstant = 0.002f;
		
	};



	struct WorkloadLayout
	{
		
		//Use Nodes to store all the data, as the destructor will only delete the Nodes pointer.
		
		unsigned* Nodes = nullptr;
		unsigned* Weights = nullptr;
		unsigned* Biases = nullptr;
		unsigned* Z = nullptr;
		unsigned* Input = nullptr;
		unsigned* Output = nullptr;

		//"Whole" arrays contains the workload layout for the entire  flattend array of its namesake, instead of layer by layer and unflattened.
		unsigned* WholeNodes = nullptr;
		unsigned* WholeWeights = nullptr;
		unsigned* WholeBiases = nullptr;
		unsigned* WholeZ = nullptr;
		unsigned* WholeInput = nullptr;
		unsigned* WholeOutput = nullptr;

		~WorkloadLayout()
		{
			delete[] Nodes;
		}
	};



	struct HyperParameters
	{
		//General
		unsigned Epochs;


		//Stochastic Gradient Decent
		unsigned BatchCount;

	};


	//ForwardDeclaration
	class NetworkPrototype;

	struct FunctionsLayout
	{
		
		struct NeuronFunction
		{
			float (*f)(float z);
		};
		struct NetworkRelayFunction
		{
			void (*f)(NetworkPrototype* n);
		};




		//NeuronFunctions and their derivatives
		NeuronFunction* NeuronFunctions;
		//Derivatives here are not in reverse order compared to layers.
		NeuronFunction* NeuronFunctionsDerivatives;



		//Functions that descripe how layers are connected and what operations are preformed on their activations.
		//Count of this array is m_LayerLayoutCount-1, where the first element applies to the zs of the second layer (no zs in the inputlayer) and the last applies to the zs of the outputlayer.
		NetworkRelayFunction* FeedForwardCallBackFunctions;

		//The output layert has its derivative (with repect to z) calculated in the CostFunctionDerivative function, so the Count of this array is m_LayerLayoutCount-2. Function for the last layer is supposed to be first in the array.
		NetworkRelayFunction* BackPropegateCallBackFunctionsZ; //Meant for calculating the entries of m_DeltaZ
		//This one has a count of m_LayerLayoutCount-1
		NetworkRelayFunction* BackPropegateCallBackFunctionsBW; //Meant for calculating the entries of m_DeltaBiases and m_DeltaWeights
		
	
		//Cost function. Its "derivative" is used as the beackpropegation function for the last layer.
		NetworkRelayFunction CostFunction;
		//Is supposed to be the derivative with respects to z in the outputlayer, effectively meaning that this function is: dC/da * da/dz.
		NetworkRelayFunction CostFunctionDerivative;


		//For the trainingprocess
		NetworkRelayFunction* TrainingFunctions;

		NetworkRelayFunction* RegularizationFunctions;




		
		~FunctionsLayout()
		{
			delete[] NeuronFunctions;
			delete[] NeuronFunctionsDerivatives;

			delete[] FeedForwardCallBackFunctions;
			delete[] BackPropegateCallBackFunctionsZ;
			delete[] BackPropegateCallBackFunctionsBW;

			delete[] TrainingFunctions;
			delete[] RegularizationFunctions;
		}

	};


	class NetworkPrototypeMT;

	struct FunctionsLayoutMT
	{

		struct NeuronFunction
		{
			float (*f)(float z);
		};
		struct NetworkRelayFunctionMT
		{
			void (*f)(NetworkPrototypeMT* n, unsigned thread);
		};




		//NeuronFunctions and their derivatives
		NeuronFunction* NeuronFunctions;
		//Derivatives here are not in reverse order compared to layers.
		NeuronFunction* NeuronFunctionsDerivatives;



		//Functions that descripe how layers are connected and what operations are preformed on their activations.
		//Count of this array is m_LayerLayoutCount-1, where the first element applies to the zs of the second layer (no zs in the inputlayer) and the last applies to the zs of the outputlayer.
		NetworkRelayFunctionMT* FeedForwardCallBackFunctions;

		//The output layert has its derivative (with repect to z) calculated in the CostFunctionDerivative function, so the Count of this array is m_LayerLayoutCount-2. 
		//Function for the last layer is supposed to be first in the array. This is because backpropegation goes from last to first layer, not first to last.
		NetworkRelayFunctionMT* BackPropegateCallBackFunctionsZ; //Meant for calculating the entries of m_DeltaZ
		//This one has a count of m_LayerLayoutCount-1
		//Function for the last layer is supposed to be first in the array. This is because backpropegation goes from last to first layer, not first to last.
		NetworkRelayFunctionMT* BackPropegateCallBackFunctionsBW; //Meant for calculating the entries of m_DeltaBiases and m_DeltaWeights


		//Cost function. Its "derivative" is used as the beackpropegation function for the last layer.
		NetworkRelayFunctionMT CostFunction;
		//Is supposed to be the derivative with respects to z in the outputlayer, effectively meaning that this function is: dC/da * da/dz.
		NetworkRelayFunctionMT CostFunctionDerivative;




		//For the trainingprocess
		NetworkRelayFunctionMT* TrainingFunctions;

		NetworkRelayFunctionMT* RegularizationFunctions;




		~FunctionsLayoutMT()
		{
			delete[] NeuronFunctions;
			delete[] NeuronFunctionsDerivatives;

			delete[] FeedForwardCallBackFunctions;
			delete[] BackPropegateCallBackFunctionsZ;
			delete[] BackPropegateCallBackFunctionsBW;

			delete[] TrainingFunctions;
			delete[] RegularizationFunctions;
		}

	};



	



	struct DataSet
	{
		float* TrainingInputs;
		float* TraningTargets;
		unsigned TrainingCount;

		float* ValidationInputs;
		float* ValidationTargets;
		unsigned ValidationCount;

		float* TestInputs;
		float* TestTargets;
		unsigned TestCount;



		~DataSet()
		{
			delete[] TrainingInputs;
			delete[] TraningTargets;

			delete[] ValidationInputs;
			delete[] ValidationTargets;

			delete[] TestInputs;
			delete[] TestTargets;
		}
	};

	struct ConditionFunctionPointer
	{
		void (*Function)(float* cost, unsigned step, unsigned& epochs, unsigned& batchSize, float& learningRate, float& regConst, bool* updateOrRevert);
	};















	

}
