#include "../../pch.h"
#include "CostFunctions.h"

float CrossEntropy(float* outputLayer, float* target, unsigned outputLayerCount)
{

	float cost = 0;

	unsigned layerIndex = 0;
	while (layerIndex < outputLayerCount)
	{

		float a = outputLayer[layerIndex];
		float y = target[layerIndex];


		cost += Math::CrossEntropy(a, y);


		layerIndex++;
	}

	return cost;

}

void CrossEntropyDerivative(float* outputLayer, float* outputZ, float* outputDZ, float* target, unsigned outputLayerCount)
{

	unsigned layerIndex = 0;
	while (layerIndex < outputLayerCount)
	{

		float z = outputZ[layerIndex];
		float a = outputLayer[layerIndex];
		float y = target[layerIndex];

		float dz = Math::CrossEntropyCostDerivative(z, a, y);

		outputDZ[layerIndex] = dz;


		layerIndex++;
	}

}