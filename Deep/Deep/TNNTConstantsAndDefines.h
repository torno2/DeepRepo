#pragma once

//Defines
#define FullyConnectedLayerDef



#define InputNeuronFunction(x) Math::Sigmoid(x)
#define InputNeuronFunctionDerivative(x) Math::SigmoidDerivative(x)

#define OutputNeuronFunction(x) Math::Sigmoid(x)
#define OutputNeuronFunctionDerivative(x) Math::SigmoidDerivative(x)


#define InputNodesCount 784
#define OutputNodesCount 10

#define Layer1NodeCount InputNodesCount
#define Layer2NodeCount 100
#define Layer3NodeCount 30
#define Layer4NodeCount OutputNodesCount



// Constants
namespace TNNT
{
	//In bytes
	constexpr unsigned CacheLineSize = 64;
}
