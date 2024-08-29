
#include "NetworkPrototype.h"


using namespace TNNT;

NetworkPrototype testNet;
int guess(float* input)
{
	return testNet.Check(input);
}