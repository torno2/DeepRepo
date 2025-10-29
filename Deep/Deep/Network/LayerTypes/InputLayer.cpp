#include "pch.h"
#include "non_pch_includes.h"
#include "InputLayer.h"


namespace TNNT {

	float* InputLayerSetup(InputLayerParams& n, float* storage, unsigned nodesCount)
	{
		n.NodesCount = nodesCount;

		n.Z = storage;
		n.A = n.Z + n.NodesCount;

		return storage;
	}

	void InputLayerTransform(InputLayerParams& n)
	{
		unsigned layerIndex = 0;
		while (layerIndex < n.NodesCount)
		{
			
		
			n.A[layerIndex] = (n.Z[layerIndex]);
			n.A[layerIndex] = InputNeuronFunction(n.Z[layerIndex]);
			layerIndex++;
		}
	}

}

