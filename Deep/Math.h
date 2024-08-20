#pragma once

namespace Math{




	float Identity(float z);
	float IdentityDerivative(float z);

	float Sigmoid(float z);
	float SigmoidDerivative(float z);



	float CrossEntropy(float a, float y);
	//Needs z arguments to be passed into neural-network
	float CrossEntropyCostDerivative(float z,float a, float y);






}