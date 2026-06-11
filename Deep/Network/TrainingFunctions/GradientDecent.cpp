#include "../../Includes/pch.h"
#include "../../Includes/non_pch_includes.h"
#include "GradientDecent.h"



namespace TNNT
{
	void L2Regularization(float* tempParameterArray, unsigned paramterCounts, unsigned trainingSetCount, float learningRate, float regConstant)
	{
		Math::ScalarMult(tempParameterArray, (1 - (learningRate * regConstant / ((float)trainingSetCount))), paramterCounts);
	}

	void GradientDecent(float* tempParameterArray, float* derivativeParameterArray, unsigned paramCount, unsigned batchSize, float learningRate )
	{


		Math::ScalarMultAdd(tempParameterArray, derivativeParameterArray, -(learningRate / ((float)batchSize)), paramCount);

	}


}

