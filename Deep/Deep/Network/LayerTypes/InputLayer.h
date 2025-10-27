#pragma once


namespace TNNT {
	struct InputLayerParams
	{

		float* Z;
		float* A;


		unsigned NodesCount;

	};

	float* InputLayerSetup(InputLayerParams& n, float* storage, unsigned nodesCount);
	void InputLayerTransform(InputLayerParams& n);
}
