#include "../../pch.h"
#include "GradientDecent.h"



namespace TNNT
{
	void L2Regularization(float* tempParameterArray, unsigned paramterCounts, float learningRate, float regConstant, unsigned trainingSetCount)
	{
		Math::ScalarMult(tempParameterArray, (1 - (learningRate * regConstant / ((float)trainingSetCount))), paramterCounts);
	}

	void GradientDecent(float* tempParameter, float* derivativeParameter, unsigned paramCount, float learningRate, unsigned batchCount)
	{


		Math::ScalarMultAdd(tempParameter, derivativeParameter, -(learningRate / ((float)batchCount)), paramCount);

	}


}

