#include "../../Includes/pch.h"
#include "../../Includes/non_pch_includes.h"
#include "CostFunctions.h"

namespace TNNT 
{



	float CrossEntropy(float* outputA, float* target, unsigned outputLayerCount)
	{

		float cost = 0;

		unsigned layerIndex = 0;
		while (layerIndex < outputLayerCount)
		{

			float a = outputA[layerIndex];
			float y = target[layerIndex];


			cost += Math::CrossEntropy(a, y);


			layerIndex++;
		}

		return cost;

	}


	void CrossEntropyDerivative(float* outputA, float* outputZ, float* outputDZ, float* target, unsigned outputLayerCount)
	{

		unsigned layerIndex = 0;
		while (layerIndex < outputLayerCount)
		{

			float z = outputZ[layerIndex];
			float a = outputA[layerIndex];
			float y = target[layerIndex];


			//This one only works if you're using the sigmoid function as a neuron function for the last layer
			float dz = Math::CrossEntropyCostDerivative(z, a, y);

			outputDZ[layerIndex] = dz;


			layerIndex++;
		}

	}

}