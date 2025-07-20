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

	size_t LCM(size_t a, size_t b)
	{

		return (a * b) / GCD(a, b);

	}

	size_t GCD(size_t a, size_t b)
	{

		while (b != 0)
		{


			size_t r = a % b;
			a = b;
			b = r;

		}
		return a;
	}




	float Dot(const float* a, const float* b, size_t count) {
		__m256 sum_vec = _mm256_setzero_ps(); // Initialize sum to zero

		size_t i = 0;
		for (; i + 7 < count; i += 8) { // Process 8 elements at a time
			__m256 va = _mm256_loadu_ps(&a[i]);
			__m256 vb = _mm256_loadu_ps(&b[i]);

			__m256 prod = _mm256_mul_ps(va, vb); // Element-wise multiplication
			sum_vec = _mm256_add_ps(sum_vec, prod); // Accumulate results
		}

		// Reduce 256-bit sum_vec into a single value
		__m128 sum_low = _mm256_extractf128_ps(sum_vec, 0);
		__m128 sum_high = _mm256_extractf128_ps(sum_vec, 1);
		__m128 sum_128 = _mm_add_ps(sum_low, sum_high); // Combine lower and upper halves

		sum_128 = _mm_hadd_ps(sum_128, sum_128); // Horizontally add pairs
		sum_128 = _mm_hadd_ps(sum_128, sum_128);

		float dot_result = _mm_cvtss_f32(sum_128); // Extract final sum

		// Handle remainder elements (if count is not a multiple of 8)
		for (; i < count; ++i) {
			dot_result += a[i] * b[i];
		}

		return dot_result;
	}


	void ScalarMult(float* vec, const float scalar, size_t count)
	{
		__m256 scalar_vec = _mm256_set1_ps(scalar);


		size_t i = 0;
		for (; i + 7 < count; i += 8) {  // Process 8 elements at a time

			__m256 tempVec = _mm256_loadu_ps(&vec[i]);

			tempVec = _mm256_mul_ps(tempVec, scalar_vec);

			_mm256_storeu_ps(&vec[i], tempVec);
		}

		// Handle remainder elements
		for (; i < count; ++i) {
			vec[i] *= scalar;
		}
	}


	void ScalarMultAdd(float* dst, float* src, float scalar, size_t count)
	{

		__m256 scalar_vec = _mm256_set1_ps(scalar);


		size_t i = 0;
		for (; i + 7 < count; i += 8) {  // Process 8 elements at a time

			__m256 mm_src = _mm256_loadu_ps(&src[i]);
			__m256 mm_dst = _mm256_loadu_ps(&dst[i]);

			mm_src = _mm256_mul_ps(mm_src, scalar_vec);
			mm_dst = _mm256_add_ps(mm_dst, mm_src);

			_mm256_storeu_ps(&dst[i], mm_dst);
		}

		// Handle remainder elements
		for (; i < count; ++i) {
			dst[i] += src[i]*scalar;
		}


	}

	void VecAdd(float* dst, float* src, size_t count)
	{
		size_t i = 0;
		for (; i + 7 < count; i += 8) {  // Process 8 elements at a time

			__m256 mm_dst = _mm256_loadu_ps(&dst[i]);
			__m256 mm_src = _mm256_loadu_ps(&src[i]);

			mm_dst = _mm256_add_ps(mm_dst, mm_src);

			_mm256_storeu_ps(&dst[i], mm_dst);
		}

		// Handle remainder elements
		for (; i < count; ++i) {
			dst[i] += src[i];
		}
	}

	void Hadamard(float* dst, float* src, size_t count)
	{


		size_t i = 0;
		for (; i + 7 < count; i += 8) {  // Process 8 elements at a time

			__m256 mm_dst = _mm256_loadu_ps(&dst[i]);
			__m256 mm_src = _mm256_loadu_ps(&src[i]);

			mm_dst = _mm256_mul_ps(mm_dst, mm_src);

			_mm256_storeu_ps(&dst[i], mm_dst);
		}

		// Handle remainder elements
		for (; i < count; ++i) {
			dst[i] *= src[i];
		}

	}








}
