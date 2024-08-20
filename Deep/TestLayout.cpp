#include "pch.h"
#include "DataProcessing.h"
#include "NetworkPrototype.h"
#include "NetworkPrototypeMT.h"
#include "LayerFunctions.h"
#include "LayerFunctionsMT.h"

#define threadN 1


#define MNIST true
#define traningCount 50000

#define train true


#define FullyConnected true
#define Convolution !FullyConnected && false 





//void Setup()
//{
//
//	using namespace TNNT;
//
//	constexpr unsigned dim = 3;
//
//	unsigned layer0Dim[dim] = { 3,3,2 };
//	unsigned ker1Dim[dim] = { 1,1, 2 };
//	unsigned stride1[dim] = { 1,1,1 };
//	unsigned subLayerCount = 1;
//
//	unsigned layerLayoutCount = 2;
//	LayerLayout* layerLayout = new LayerLayout[layerLayoutCount];
//	{
//		layerLayout[0].NodesCount = 1;
//		for (unsigned i = 0; i < dim; i++)
//		{
//			layerLayout[0].NodesCount *= layer0Dim[i];
//		}
//
//		layerLayout[0].ZCount = 0;
//		layerLayout[0].WeightsCount = 0;
//		layerLayout[0].BiasesCount = 0;
//
//		layerLayout[0].LayerDim = layer0Dim;
//		layerLayout[0].LayerDimCount = dim;
//
//
//		layerLayout[1].SubLayerCount = 1;
//
//
//		layerLayout[1].NodesCount = 1;
//		for (unsigned i = 0; i < dim; i++)
//		{
//			layerLayout[1].NodesCount *= (1 + ((layer0Dim[i] - (ker1Dim[i])) / stride1[i]));
//		}
//		layerLayout[1].NodesCount *= layerLayout[1].SubLayerCount;
//
//		layerLayout[1].ZCount = layerLayout[1].NodesCount;
//		layerLayout[1].WeightsCount = 1;
//		for (unsigned i = 0; i < dim; i++)
//		{
//			layerLayout[1].WeightsCount *= ker1Dim[i];
//		}
//		layerLayout[1].WeightsCount *= layerLayout[1].SubLayerCount;
//
//		layerLayout[1].BiasesCount = 1 * layerLayout[1].SubLayerCount;
//
//
//
//		layerLayout[1].KerDim = ker1Dim;
//		layerLayout[1].Stride = stride1;
//		layerLayout[1].KerDimCount = dim;
//
//
//	}
//
//	FunctionsLayout funcLayout;
//	{
//		funcLayout.NeuronFunctions = new FunctionsLayout::NeuronFunction[layerLayoutCount - 1];
//		{
//			funcLayout.NeuronFunctions[0].f = Math::Identity;
//		}
//		funcLayout.NeuronFunctionsDerivatives = new FunctionsLayout::NeuronFunction[layerLayoutCount - 1];
//		{
//			funcLayout.NeuronFunctionsDerivatives[0].f = Math::IdentityDerivative;
//		}
//
//		funcLayout.FeedForwardCallBackFunctions = new FunctionsLayout::NetworkRelayFunction[layerLayoutCount - 1];
//		{
//			funcLayout.FeedForwardCallBackFunctions[0].f = LayerFunctions::ConvolutionLayerFeedForward;
//		}
//
//		funcLayout.BackPropegateCallBackFunctionsZ = new FunctionsLayout::NetworkRelayFunction[layerLayoutCount - 1];
//		{
//			funcLayout.BackPropegateCallBackFunctionsZ[0].f = CostFunctions::EmptyCostFunction;
//		}
//
//		funcLayout.BackPropegateCallBackFunctionsBW = new FunctionsLayout::NetworkRelayFunction[layerLayoutCount - 1];
//		{
//			funcLayout.BackPropegateCallBackFunctionsBW[0].f = CostFunctions::EmptyCostFunction;
//		}
//
//		funcLayout.RegularizationFunctions = new FunctionsLayout::NetworkRelayFunction[layerLayoutCount - 1];
//		{
//			funcLayout.RegularizationFunctions[0].f = CostFunctions::EmptyCostFunction;
//		}
//
//		funcLayout.TrainingFunctions = new FunctionsLayout::NetworkRelayFunction[layerLayoutCount - 1];
//		{
//			funcLayout.TrainingFunctions[0].f = CostFunctions::EmptyCostFunction;
//		}
//		funcLayout.CostFunction.f = CostFunctions::EmptyCostFunction;
//		funcLayout.CostFunctionDerivative.f = CostFunctions::EmptyCostFunction;
//
//	}
//
//
//
//	NetworkPrototype n(layerLayout, funcLayout, layerLayoutCount, false);
//
//
//
//
//	{
//
//
//
//		using namespace TNNT;
//		Timer t;
//
//		//DataFormating start
//		t.Start();
//
//
//
//#if MNIST
//
//		DataSet data;
//		{
//			constexpr unsigned labelSize = 10;
//			constexpr unsigned inputSize = 28 * 28;
//
//			data.TrainingCount = traningCount;
//			data.TrainingInputs = new float[data.TrainingCount * inputSize];
//			data.TraningTargets = new float[data.TrainingCount * labelSize];
//
//			data.ValidationCount = 10000;
//			data.ValidationInputs = new float[data.ValidationCount * inputSize];
//			data.ValidationTargets = new float[data.ValidationCount * labelSize];
//
//			data.TestCount = 10000;
//			data.TestInputs = new float[data.TestCount * inputSize];
//			data.TestTargets = new float[data.TestCount * labelSize];
//
//
//			//Data Formating start
//
//			ProcessMNISTDataMT(10, data.TrainingInputs, data.TraningTargets, "trainLabel.idx1-ubyte", "trainIm.idx3-ubyte", data.TrainingCount);
//			ProcessMNISTDataMT(10, data.ValidationInputs, data.ValidationTargets, "trainLabel.idx1-ubyte", "trainIm.idx3-ubyte", data.ValidationCount, 50000);
//			ProcessMNISTDataMT(10, data.TestInputs, data.TestTargets, "testLabel.idx1-ubyte", "testIm.idx3-ubyte", data.TestCount);
//		}
//#endif 
//
//
//		pr("Data formating time:" << t.Stop() << "s");
//
//
//
//
//		//Prototype Network setup start
//
//
//#if FullyConnected
//
//		unsigned int fullyConnectedLayoutCount = 4;
//		LayerLayout* fullyConnectedLayout = new LayerLayout[fullyConnectedLayoutCount];
//		{
//			fullyConnectedLayout[0].NodesCount = 28 * 28;
//			fullyConnectedLayout[0].BiasesCount = 0;
//			fullyConnectedLayout[0].WeightsCount = 0;
//
//			fullyConnectedLayout[1].NodesCount = 30;
//			fullyConnectedLayout[1].BiasesCount = fullyConnectedLayout[1].NodesCount;
//			fullyConnectedLayout[1].WeightsCount = fullyConnectedLayout[1].NodesCount * fullyConnectedLayout[1 - 1].NodesCount;
//
//			fullyConnectedLayout[2].NodesCount = 30;
//			fullyConnectedLayout[2].BiasesCount = fullyConnectedLayout[2].NodesCount;
//			fullyConnectedLayout[2].WeightsCount = fullyConnectedLayout[2].NodesCount * fullyConnectedLayout[2 - 1].NodesCount;
//
//			fullyConnectedLayout[3].NodesCount = 10;
//			fullyConnectedLayout[3].BiasesCount = fullyConnectedLayout[3].NodesCount;
//			fullyConnectedLayout[3].WeightsCount = fullyConnectedLayout[3].NodesCount * fullyConnectedLayout[3 - 1].NodesCount;
//
//
//		}
//
//		FunctionsLayout fullyConnectedFuncLayout;
//		{
//
//			fullyConnectedFuncLayout.NeuronFunctions = new FunctionsLayout::NeuronFunction[fullyConnectedLayoutCount - 1];
//			{
//				fullyConnectedFuncLayout.NeuronFunctions[0].f = Math::Sigmoid;
//				fullyConnectedFuncLayout.NeuronFunctions[1].f = Math::Sigmoid;
//				fullyConnectedFuncLayout.NeuronFunctions[2].f = Math::Sigmoid;
//				;
//			}
//			fullyConnectedFuncLayout.NeuronFunctionsDerivatives = new FunctionsLayout::NeuronFunction[fullyConnectedLayoutCount - 1];
//			{
//				fullyConnectedFuncLayout.NeuronFunctionsDerivatives[0].f = Math::SigmoidDerivative;
//				fullyConnectedFuncLayout.NeuronFunctionsDerivatives[1].f = Math::SigmoidDerivative;
//				fullyConnectedFuncLayout.NeuronFunctionsDerivatives[2].f = Math::SigmoidDerivative;
//
//			}
//
//			fullyConnectedFuncLayout.FeedForwardCallBackFunctions = new FunctionsLayout::NetworkRelayFunction[fullyConnectedLayoutCount - 1];
//			{
//				fullyConnectedFuncLayout.FeedForwardCallBackFunctions[0].f = LayerFunctions::FullyConnectedFeedForward;
//				fullyConnectedFuncLayout.FeedForwardCallBackFunctions[1].f = LayerFunctions::FullyConnectedFeedForward;
//				fullyConnectedFuncLayout.FeedForwardCallBackFunctions[2].f = LayerFunctions::FullyConnectedFeedForward;
//
//			}
//
//			fullyConnectedFuncLayout.BackPropegateCallBackFunctionsZ = new FunctionsLayout::NetworkRelayFunction[fullyConnectedLayoutCount - 1];
//			{
//				fullyConnectedFuncLayout.BackPropegateCallBackFunctionsZ[0].f = LayerFunctions::FullyConnectedBackpropegateZ;
//				fullyConnectedFuncLayout.BackPropegateCallBackFunctionsZ[1].f = LayerFunctions::FullyConnectedBackpropegateZ;
//				fullyConnectedFuncLayout.BackPropegateCallBackFunctionsZ[2].f = CostFunctions::CrossEntropyDerivative;
//
//
//			}
//
//			fullyConnectedFuncLayout.BackPropegateCallBackFunctionsBW = new FunctionsLayout::NetworkRelayFunction[fullyConnectedLayoutCount - 1];
//			{
//				fullyConnectedFuncLayout.BackPropegateCallBackFunctionsBW[0].f = LayerFunctions::FullyConnectedBackpropegateBW;
//				fullyConnectedFuncLayout.BackPropegateCallBackFunctionsBW[1].f = LayerFunctions::FullyConnectedBackpropegateBW;
//				fullyConnectedFuncLayout.BackPropegateCallBackFunctionsBW[2].f = LayerFunctions::FullyConnectedBackpropegateBW;
//			}
//
//
//
//			fullyConnectedFuncLayout.CostFunction.f = CostFunctions::CrossEntropy;
//			fullyConnectedFuncLayout.CostFunctionDerivative.f = CostFunctions::CrossEntropyDerivative;
//
//
//
//			fullyConnectedFuncLayout.RegularizationFunctions = new FunctionsLayout::NetworkRelayFunction[fullyConnectedLayoutCount - 1];
//			{
//				fullyConnectedFuncLayout.RegularizationFunctions[0].f = TrainingFunctions::L2Regularization;
//				fullyConnectedFuncLayout.RegularizationFunctions[1].f = TrainingFunctions::L2Regularization;
//				fullyConnectedFuncLayout.RegularizationFunctions[2].f = TrainingFunctions::L2Regularization;
//
//			}
//
//
//			fullyConnectedFuncLayout.TrainingFunctions = new FunctionsLayout::NetworkRelayFunction[fullyConnectedLayoutCount - 1];
//			{
//				fullyConnectedFuncLayout.TrainingFunctions[0].f = TrainingFunctions::GradientDecent;
//				fullyConnectedFuncLayout.TrainingFunctions[1].f = TrainingFunctions::GradientDecent;
//				fullyConnectedFuncLayout.TrainingFunctions[2].f = TrainingFunctions::GradientDecent;
//
//			}
//		}
//
//#endif
//
//
//
//		NetworkPrototype nP(fullyConnectedLayout, fullyConnectedFuncLayout, fullyConnectedLayoutCount, true);
//
//		//Prototype Network setup stop
//
//
//
//		// MT prototype setup start
//#if FullyConnected
//		unsigned int MTlayoutCount = 4;
//		LayerLayout* MTlayout = new LayerLayout[MTlayoutCount];
//		{
//			MTlayout[0].NodesCount = 28 * 28;
//			MTlayout[0].ZCount = 0;
//			MTlayout[0].BiasesCount = 0;
//			MTlayout[0].WeightsCount = 0;
//
//			MTlayout[1].NodesCount = 30;
//			MTlayout[1].ZCount = 30;
//			MTlayout[1].BiasesCount = MTlayout[1].NodesCount;
//			MTlayout[1].WeightsCount = MTlayout[1].NodesCount * MTlayout[0].NodesCount;
//
//			MTlayout[2].NodesCount = 30;
//			MTlayout[2].ZCount = 30;
//			MTlayout[2].BiasesCount = MTlayout[2].NodesCount;
//			MTlayout[2].WeightsCount = MTlayout[2].NodesCount * MTlayout[1].NodesCount;
//
//
//			MTlayout[3].NodesCount = 10;
//			MTlayout[3].ZCount = 10;
//			MTlayout[3].BiasesCount = MTlayout[3].NodesCount;
//			MTlayout[3].WeightsCount = MTlayout[3].NodesCount * MTlayout[2].NodesCount;
//
//
//		}
//
//		FunctionsLayoutMT mtFuncLayout;
//		{
//
//			mtFuncLayout.NeuronFunctions = new FunctionsLayoutMT::NeuronFunction[MTlayoutCount - 1];
//			{
//				mtFuncLayout.NeuronFunctions[0].f = Math::Sigmoid;
//				mtFuncLayout.NeuronFunctions[1].f = Math::Sigmoid;
//				mtFuncLayout.NeuronFunctions[2].f = Math::Sigmoid;
//				;
//			}
//			mtFuncLayout.NeuronFunctionsDerivatives = new FunctionsLayoutMT::NeuronFunction[MTlayoutCount - 1];
//			{
//				mtFuncLayout.NeuronFunctionsDerivatives[0].f = Math::SigmoidDerivative;
//				mtFuncLayout.NeuronFunctionsDerivatives[1].f = Math::SigmoidDerivative;
//				mtFuncLayout.NeuronFunctionsDerivatives[2].f = Math::SigmoidDerivative;
//
//			}
//
//			mtFuncLayout.FeedForwardCallBackFunctions = new FunctionsLayoutMT::NetworkRelayFunctionMT[MTlayoutCount - 1];
//			{
//				mtFuncLayout.FeedForwardCallBackFunctions[0].f = LayerFunctionsMT::FullyConnectedFeedForward;
//				mtFuncLayout.FeedForwardCallBackFunctions[1].f = LayerFunctionsMT::FullyConnectedFeedForward;
//				mtFuncLayout.FeedForwardCallBackFunctions[2].f = LayerFunctionsMT::FullyConnectedFeedForward;
//
//			}
//
//			mtFuncLayout.BackPropegateCallBackFunctionsZ = new FunctionsLayoutMT::NetworkRelayFunctionMT[MTlayoutCount - 1];
//			{
//
//				mtFuncLayout.BackPropegateCallBackFunctionsZ[0].f = LayerFunctionsMT::FullyConnectedBackpropegateZ;
//				mtFuncLayout.BackPropegateCallBackFunctionsZ[1].f = LayerFunctionsMT::FullyConnectedBackpropegateZ;
//				mtFuncLayout.BackPropegateCallBackFunctionsZ[2].f = CostFunctionsMT::CrossEntropyDerivative;
//
//			}
//
//			mtFuncLayout.BackPropegateCallBackFunctionsBW = new FunctionsLayoutMT::NetworkRelayFunctionMT[MTlayoutCount - 1];
//			{
//
//				mtFuncLayout.BackPropegateCallBackFunctionsBW[0].f = LayerFunctionsMT::FullyConnectedBackpropegateBW;
//				mtFuncLayout.BackPropegateCallBackFunctionsBW[1].f = LayerFunctionsMT::FullyConnectedBackpropegateBW;
//				mtFuncLayout.BackPropegateCallBackFunctionsBW[2].f = LayerFunctionsMT::FullyConnectedBackpropegateBW;
//			}
//
//
//
//			mtFuncLayout.CostFunction.f = CostFunctionsMT::CrossEntropy;
//			mtFuncLayout.CostFunctionDerivative.f = CostFunctionsMT::CrossEntropyDerivative;
//
//
//			mtFuncLayout.RegularizationFunctions = new FunctionsLayoutMT::NetworkRelayFunctionMT[MTlayoutCount - 1];
//			{
//				mtFuncLayout.RegularizationFunctions[0].f = TrainingFunctionsMT::L2Regularization;
//				mtFuncLayout.RegularizationFunctions[1].f = TrainingFunctionsMT::L2Regularization;
//				mtFuncLayout.RegularizationFunctions[2].f = TrainingFunctionsMT::L2Regularization;
//			}
//
//			mtFuncLayout.TrainingFunctions = new FunctionsLayoutMT::NetworkRelayFunctionMT[MTlayoutCount - 1];
//
//			{
//				mtFuncLayout.TrainingFunctions[0].f = TrainingFunctionsMT::GradientDecent;
//				mtFuncLayout.TrainingFunctions[1].f = TrainingFunctionsMT::GradientDecent;
//				mtFuncLayout.TrainingFunctions[2].f = TrainingFunctionsMT::GradientDecent;
//			}
//
//		}
//
//		NetworkPrototypeMT nMT(MTlayout, mtFuncLayout, MTlayoutCount, threadN, true);
//		// MT prototype setup stop
//
//#endif
//
//
//		HyperParameters params;
//		{
//			params.Epochs = 1;
//
//			params.BatchCount = 9;
//
//		}
//
//
//		HyperParameters paramsMT;
//		{
//			paramsMT.Epochs = 1;
//
//			paramsMT.BatchCount = 9;
//
//		}
//
//
//
//	}
//}


