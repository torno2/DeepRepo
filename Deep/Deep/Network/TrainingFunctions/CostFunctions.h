#pragma once

namespace TNNT
{

	float CrossEntropy(float* outputA, float* target, unsigned outputLayerCount);

	void CrossEntropyDerivative(float* outputA, float* outputZ, float* outputDZ, float* target, unsigned outputLayerCount);
}