#include "pch.h"
#include "Math.h"

namespace Math
{
	float Identity(float z)
	{
		return z;
	}
	float IdentityDerivative(float z)
	{
		return 1;
	}
	float Sigmoid(float z)
	{
		
		float result = 1 / (1 + std::exp(-z));
		return result;
	}

	float SigmoidDerivative(float z)
	{
		float result = Sigmoid(z) * (1 - Sigmoid(z));
		return result;
	}

	float CrossEntropy(float a, float y)
	{
		float result;
		if ( ((a==1) && (y==1)) || ((a == 0) && (y == 0)) )
		{
			
			result = 0;
		}
		//else if (((a == 1) && (y == 0)) || ((a ==0) && (y == 1)))
		//{
		//	result = std::numeric_limits<float>::max();
		//}
		else {
			result = -((y * std::log(a)) + ((1 - y) * std::log(1 - a)));
		}
		
		return result;
	}
	// dC/da * da/dz where a = sigmoid(z)
	float CrossEntropyCostDerivative(float z, float a, float y)
	{
		return(a - y);
	}













}
