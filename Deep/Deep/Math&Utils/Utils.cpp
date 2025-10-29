#include "pch.h"
#include "non_pch_includes.h"
#include "Utils.h"

void PrintWeights(unsigned* layout, unsigned layoutSize, float* weights,unsigned layer)
{
	if (layer == 0)
	{
		unsigned weightsStart = 0;
		for (unsigned layoutIndex = 1; layoutIndex < layoutSize; layoutIndex++)
		{
			
			for (unsigned layerIndex = 0; layerIndex < layout[layoutIndex]; layerIndex++)
			{

				for (unsigned prevLayer = 0; prevLayer < layout[layoutIndex - 1]; prevLayer++)
				{

					std::cout << " |" << prevLayer + layout[layoutIndex - 1] * layerIndex << ": " << weights[weightsStart+prevLayer + layout[layoutIndex - 1] * layerIndex];

				}

				std::cout << std::endl;
			}
			weightsStart += layout[layoutIndex - 1] * layout[layoutIndex];
		}
	}
	else
	{
		unsigned weightsStart = 0;
		for (unsigned layoutIndex = 1; layoutIndex < layoutSize; layoutIndex++)
		{
			if (layer == layoutIndex)
			{
				for (unsigned layerIndex = 0; layerIndex < layout[layoutIndex]; layerIndex++)
				{

					for (unsigned prevLayer = 0; prevLayer < layout[layoutIndex - 1]; prevLayer++)
					{

						std::cout << " |" << prevLayer + layout[layoutIndex - 1] * layerIndex << ": " << weights[weightsStart + prevLayer + layout[layoutIndex - 1] * layerIndex];

					}

					std::cout << std::endl;
				}
			}
			weightsStart += layout[layoutIndex - 1] * layout[layoutIndex];
		}
	}
}

void ShuffleFloat(float* arr, std::mt19937& mersenneGenerator, unsigned count)
{
	unsigned i = 0;
	unsigned randomIndexCount = count;
	while (i < count - 1)
	{
		unsigned randomIndex = (mersenneGenerator() % randomIndexCount) + i;

		float buffer = arr[randomIndex];
		arr[randomIndex] = arr[i];
		arr[i] = buffer;

		i++;
		randomIndexCount--;
	}
}

void ShuffleInt(unsigned* arr, std::mt19937& mersenneGenerator, unsigned count)
{
	unsigned i = 0;
	unsigned randomIndexCount = count;
	while (i < count - 1)
	{
		unsigned randomIndex = (mersenneGenerator() % randomIndexCount) + i;

		unsigned buffer = arr[randomIndex];
		arr[randomIndex] = arr[i];
		arr[i] = buffer;

		i++;
		randomIndexCount--;
	}
}

//For shuffeling "count" tuples of size "elementsize"
void ShuffleFloatTuples(float* arr, float* buffer,  std::mt19937& mersenneGenerator, unsigned count, unsigned elementsize)
{


	unsigned i = 0;
	unsigned randomIndexCount = count;
	while (i < count-1)
	{
		unsigned randomIndex = (mersenneGenerator() % randomIndexCount) + i;

		memcpy(buffer,&arr[elementsize *randomIndex], elementsize);
		memcpy(&arr[elementsize * randomIndex],&arr[elementsize * i], elementsize);
		memcpy(&arr[elementsize * i], buffer, elementsize);

		i++;
		randomIndexCount--;
	}
}

//For shuffeling "count" tuples of size "elementsize"
void ShuffleIntTuples(unsigned* arr, unsigned* buffer, std::mt19937& mersenneGenerator, unsigned count, unsigned elementsize)
{

	unsigned i = 0;
	unsigned randomIndexCount = count;
	while (i < count - 1)
	{
		unsigned randomIndex = (mersenneGenerator() % randomIndexCount) + i;

		memcpy(buffer, &arr[elementsize * randomIndex], elementsize);
		memcpy(&arr[elementsize * randomIndex], &arr[elementsize * i], elementsize);
		memcpy(&arr[elementsize * i], buffer, elementsize);

		i++;
		randomIndexCount--;
	}
}

void PrintImg(float* img, unsigned width, unsigned height)
{
	for (int j = 0; j < height; j++)
	{
		for (int i = 0; i < width; i++)
		{
			if (img[j * width + i] == 0)
			{
				std::cout << " ";
			}
			else
			{
				std::cout << "x";
			}
			
		}
		pr("");
	}


}

void PrintMat(float* mat, unsigned width, unsigned height)
{
	for (int j = 0; j < height; j++)
	{
		for (int i = 0; i < width; i++)
		{
			std::cout << mat[i + j * width] << " ";

		}
		pr("");
	}


}

void Timer::Start()
{
	start = std::chrono::high_resolution_clock::now();
}

float Timer::Stop()
{
	auto stop = std::chrono::high_resolution_clock::now();
	std::chrono::duration<float>  time = stop - start;
	return time.count();
}



void ThreadWorkloadDividerUtils(unsigned& start, unsigned& stop, unsigned workCount, unsigned thread, unsigned threadCount)
{
	start = 0;
	stop = 0;

	unsigned workloadCount = (workCount / threadCount);
	unsigned workloadRemainder = (workCount % threadCount);


	if (thread < workloadRemainder)
	{
		start = (workloadCount + 1) * thread;
		stop = start + (workloadCount + 1);
	}
	else
	{
		start = workloadCount * thread + workloadRemainder;

		stop = start + workloadCount;
	}
}

void ThreadWorkloadDividerWithPaddingUtils(unsigned& start, unsigned& stop, unsigned workCount, unsigned thread, unsigned threadCount, unsigned padding)
{
	start = 0;
	stop = 0;

	unsigned workloadCount = (workCount / threadCount);
	unsigned workloadRemainder = (workCount % threadCount);


	if (thread < workloadRemainder)
	{
		start = (workloadCount + 1) * thread + padding * thread;
		stop = start + (workloadCount + 1);
	}
	else
	{
		start = workloadCount * thread + workloadRemainder + padding * thread;

		stop = start + workloadCount;
	}
}

void* allocate_aligned(size_t size, size_t alignment)
{
	return _aligned_malloc(size, alignment);
}

void deallocate_aligned(void* ptr)
{
	_aligned_free(ptr);
}
