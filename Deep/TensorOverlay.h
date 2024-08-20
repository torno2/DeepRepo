#pragma once

namespace TNNT
{
	struct TensorOverlay
	{
		unsigned DataCount;
		float* Data;


		unsigned DataDimCount;
		unsigned* DataDim;



		TensorOverlay(float* data, unsigned dataCount, unsigned* dataDim, unsigned dataDimCount) : Data(data), DataCount(dataCount), DataDimCount(dataDimCount), DataDim(dataDim)
		{}


		TensorOverlay(float* data, unsigned* dataDim, unsigned dataDimCount) : Data(data), DataDimCount(dataDimCount), DataDim(dataDim)
		{
			DataCount = 1;
			unsigned i = 0;
			while( i < DataDimCount)
			{
				DataCount *= DataDim[i];
				i++;
			}

		}


		TensorOverlay(float* data, unsigned dataDim, unsigned dataDimCount) : Data(data), DataDimCount(dataDimCount)
		{
			DataCount = 1;
			unsigned i = 0;
			while (i < DataDimCount)
			{
				DataDim[i] = dataDim;
				DataCount *= dataDim;
				i++;
			}

		}


		unsigned const Pos(unsigned* pos)
		{
			unsigned realPos = 0;
			unsigned mult = 1;

			unsigned i = 0;
			while (i < DataDimCount)
			{
				realPos += (pos[i] * mult );
				mult *= DataDim[i];
				i++;
			}

			if (realPos > DataCount)
			{
				printf("Error, exceeds the bounds of the tensor");
			}

			return realPos;
		}

		void const rPos(unsigned realPos, unsigned* buffer)
		{
			unsigned Div = 1;

			unsigned i = 0;
			while (i < DataDimCount)
			{
				Div *= DataDim[i];
				buffer[i] = (realPos % Div);
				realPos -= buffer[i];
				i++;
			}

			if (realPos > DataCount)
			{
				printf("Error, exceeds the bounds of the tensor");
			}
		}

		float* At(unsigned* pos)
		{
			unsigned realPos = 0;
			unsigned mult = 1;

			unsigned i = 0;
			while (i < DataDimCount)
			{
				realPos += (pos[i] * mult);
				mult *= DataDim[i];
				i++;
			}

			if (realPos > DataCount)
			{
				printf("Error, exceeds the bounds of the tensor");
			}

			return &Data[realPos];
		}




	};
}
