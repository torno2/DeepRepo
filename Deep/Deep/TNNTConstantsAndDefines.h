#pragma once

//Defines
#define FullyConnectedLayerDef



#define InputNeuronFunction(x) Math::Sigmoid(x)
#define InputNeuronFunctionDerivative(x) Math::SigmoidDerivative(x)

#define OutputNeuronFunction(x) Math::Sigmoid(x)
#define OutputNeuronFunctionDerivative(x) Math::SigmoidDerivative(x)


#define InputNodesCount 784
#define OutputNodesCount 10


// Constants
namespace TNNT
{
	//In bytes
	constexpr unsigned CacheLineSize = 64;
}
