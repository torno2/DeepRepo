#pragma once


namespace TNNT
{
	void L2Regularization(float* tempParameterArray, unsigned paramterCounts, unsigned trainingSetCount, float learningRate = 0.01f, float regConstant = 0.001f );

	void GradientDecent(float* tempParameterArray, float* derivativeParameterArray, unsigned paramCount, unsigned batchSize, float learningRate = 0.01f);
}
