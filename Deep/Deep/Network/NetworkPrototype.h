#pragma once

#include "../Control.h"



namespace TNNT {


	struct TrainingMetaData
	{
		float* TrainingDataInputs;
		float* TrainingDataTargets;
		
		unsigned* TrainingIndices;

		unsigned InputSize = 784;
		unsigned TargetSize = 10;

		unsigned TrainingCount = 50000;

		unsigned Epochs = 60;
		unsigned BatchSize = 10;

		float LearningRate = 0.1f;
		float RegConstant = 0.001f;


	};

	

	void SetInput(float* input);
	void SetTarget(float* target);
	void PreParameterAdjustment(TrainingMetaData m);
	void ParameterAdjustment(TrainingMetaData m, unsigned actualBatchSize);
	void PostParameterAdjustment();


	void SetupNetwork(float* trainingData, float* trainingTargets, unsigned trainingCount);
	void FeedForwardPass();
	void BackpropegationPass(float* target);
	
	void TrainOnSet(TrainingMetaData m, unsigned actualBatchSize);
	void TrainingLoop();

	void TestNetwork(float* testData, float* testTargets, unsigned testCount);

}