#pragma once


namespace TNNT
{
	class NetworkPrototype;

	typedef float (*NeuronFunc)(float);
	typedef void (*NetworkRelayFunction)(NetworkPrototype*);



	struct LayerHeader
	{

		void (*Setup)(char*, unsigned alignment);
		//SetupFunction Setup;

		NeuronFunc NeuronFunction;
		NeuronFunc MeuronFunctionDerivative;

		NetworkRelayFunction FeedForward;
		NetworkRelayFunction BackPropagateZ;
		NetworkRelayFunction BackPropagateBW;

		NetworkRelayFunction TrainingFunctions;

		NetworkRelayFunction RegularizationFunctions;

		float* A;

		float* Z;
		float* dZ;




		//unsigned long long storageSize;
		unsigned long long parentSize;

		unsigned NodesCount;
		unsigned ZCount;


		float LearningRate = 0.01f;
		float RegularizationConstant = 0.001f;


	};





	struct FullyConnectedLayer
	{



		float* Z;
		float* dZ;

		float* A;

		float* Weights;
		float* Biases;


		unsigned NodesCount;

		unsigned BiasesCount;
		unsigned WeightsCount;




	};

	void FeedForward_FullyConnected();

	void FeedForward_BackPropegation_FullyConnected();
	void BackPropegation_Z_FullyConnected();
	void BackPropegation_BW_FullyConnected();

	struct FullyConnectedLayerBP
	{



		float* Z;
		float* dZ;

		float* A;

		float* Weights;
		float* Biases;

		float* TempWeights;
		float* TempBiases;

		float* dWeights;
		float* dBiases;

		float* WeightsTranspose;
		unsigned Tm;

		unsigned NodesCount;

		unsigned BiasesCount;
		unsigned WeightsCount;





	};

	struct FullyConnectedLayerTR
	{



		float* Z;
		float* dZ;

		float* A;

		float* Weights;
		float* Biases;

		float* TempWeights;
		float* TempBiases;

		float* dWeights;
		float* dBiases;

		float* WeightsTranspose;
		unsigned Tm;

		unsigned NodesCount;

		unsigned BiasesCount;
		unsigned WeightsCount;




		void Setup(char* storage);



		void FeedForward(FullyConnectedLayer* n);



		void ResetTranspose();

	};













	// Example of Convolutional Layer
	struct ConvolutionalLayer
	{
		LayerHeader header;

		// --- Your layer-specific data ---

		unsigned* kerDim;
		unsigned* Stride;
		unsigned* Padding;
		unsigned kerDimCount;

		unsigned ChannelsIn;
		unsigned ChannelsOut;


	};



}

