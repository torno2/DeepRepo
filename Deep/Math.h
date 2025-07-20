#pragma once

namespace Math{




	float Identity(float z);
	float IdentityDerivative(float z);

	float Sigmoid(float z);
	float SigmoidDerivative(float z);



	float CrossEntropy(float a, float y);
	//Needs z arguments to be passed into neural-network
	float CrossEntropyCostDerivative(float z,float a, float y);

	size_t LCM(size_t a, size_t b);
	size_t GCD(size_t a, size_t b);
	
	float Dot(const float* a, const float* b, size_t count);

	void ScalarMult(float* vec, const float scalar, size_t count);
	void ScalarMultAdd(float* dst, float* src, float scalar, size_t count); //Scalar multiplies src with scalar, and then adds the result to dst.

	void VecAdd(float* dst, float* src, size_t count);

	void Hadamard(float* dst, float* src, size_t count);

	




}