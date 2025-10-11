#pragma once


namespace TNNT
{
	void L2Regularization(float* tempParameter, unsigned paramterCounts, float learningRate, float regConstant, unsigned trainingSetCount);

	void GradientDecent(float* tempParameter, float* derivativeParameter, unsigned paramCount, float learningRate, unsigned batchCount);
}
