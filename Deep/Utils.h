#pragma once
#include <iostream>
#include <fstream>

#define pr(x) std::cout << x <<'\n'

template <typename T>
std::ostream& operator<<(std::ostream& os, const std::vector<T>& v)
{
	os << "Vector Start" << std::endl;
	for (int i = 0; i < v.size(); i++)
	{
		os << i << " : " << v[i] << std::endl;
	}
	os << "Vector End";
	return os;
}

void PrintWeights(unsigned* layout, unsigned layoutSize, float* weights,unsigned layer = 0);

template <typename T>
void PArr(T* arr, unsigned count)
{
	for (int i = 0; i < count; i++)
	{
		std::cout << arr[i] << " ";
	}
	std::cout << std::endl;
}

template <typename T>
unsigned ArrayThresholdViolationCheck(T* arr, unsigned count, T lower, T upper)
{
	unsigned counter = 0;
	for (unsigned i = 0; i < count; i++)
	{
		if (arr[i] > upper || arr[i] < lower)
		{
			counter++;
		}
	}

	return counter;
}

template <typename T>
unsigned ArrayMatchCheck(T* arr, unsigned count, T target)
{
	unsigned counter = 0;
	for (unsigned i = 0; i < count; i++)
	{
		if (arr[i] == target)
		{
			counter++;
		}
	}

	return counter;
}

template <typename T>
unsigned ArrayMatchArrayCheck(T* arr1, T* arr2, unsigned count)
{
	unsigned counter = 0;
	for (unsigned index = 0; index < count; index++)
	{
		T temp1 = arr1[index];
		T temp2 = arr2[index];



		if (temp1 == temp2)
		{
			counter++;
		}
		else
		{
			if (abs(temp1 - temp2) > 200.0f)
			{
				auto t = 1;
			}
		}
	}

	return counter;
}



template <typename T>
unsigned ArrayMissmatchArrayCheck(T* arr1, T* arr2, unsigned count)
{
	unsigned counter = 0;
	for (unsigned i = 0; i < count; i++)
	{
		if (arr1[i] != arr2[i])
		{
			counter++;
		}
	}

	return counter;
}


struct Timer
{
private:
	float start;
public:
	void Start();
	float Stop();
};

void PrintImg(float* img, unsigned width, unsigned height);
void PrintMat(float* mat, unsigned width, unsigned height);



void ThreadWorkloadDividerUtils(unsigned& start, unsigned& stop, unsigned workCount, unsigned thread, unsigned threadCount);

void ThreadWorkloadDividerWithPaddingUtils(unsigned& start, unsigned& stop, unsigned workCount, unsigned thread, unsigned threadCount , unsigned padding);

template <typename T>
void LogChanges(T* target, bool* stopToken)
{
	std::ofstream log("Experiment.txt");

	std::vector<float> logBuff;
	T prev = *target;
	
	while (*stopToken)
	{
		if (*target != prev)
		{
			logBuff.push_back(prev);

			prev = *target;
			
		}

	}

	logBuff.push_back(prev);

	unsigned i = 0;
	while (i<logBuff.size())
	{



		log << i << ": " << prev << '\n';
			
		
		i++;
	}


	log.close();
}

