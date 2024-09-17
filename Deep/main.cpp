#include "pch.h"
#include "DataProcessing.h"
#include "NetworkPrototype.h"
#include "NetworkPrototype2.h"
//#include "NetworkPrototypeMT.h"
#include "LayerFunctions.h"
//#include "LayerFunctionsMT.h"
//#include "webuse.cpp"

#define test false
#define testConv false

#define clean false
#define setup true
#define threadN 1


#define MNIST true
#define traningCount 50000

#define train true


#define FullyConnected true
#define Convolution !FullyConnected && false 

#define FullyConnectedMT false
#define ConvolutionMT !FullyConnectedMT && false 

#define testPerformance true
#define testPrototype testPerformance && true
#define testMultithread testPerformance && false








int main() {
	using namespace TNNT;















#if clean == false
	{
#if testConv
		constexpr unsigned dim = 3;

		unsigned layer0Dim[dim] = { 3,3,2 };
		unsigned ker1Dim[dim] = { 1,1, 1 };
		unsigned stride1[dim] = { 1,1,1 };
		unsigned subLayerCount = 1;

		unsigned layerLayoutCount = 2;
		LayerLayout* layerLayout = new LayerLayout[layerLayoutCount];
		{
			layerLayout[0].NodesCount = 1;
			for (unsigned i = 0; i < dim; i++)
			{
				layerLayout[0].NodesCount *= layer0Dim[i];
			}

			layerLayout[0].ZCount = 0;
			layerLayout[0].WeightsCount = 0;
			layerLayout[0].BiasesCount = 0;

			layerLayout[0].LayerDim = layer0Dim;
			layerLayout[0].LayerDimCount = dim;


			layerLayout[1].SubLayerCount = 1;


			layerLayout[1].NodesCount = 1;
			for (unsigned i = 0; i < dim; i++)
			{
				layerLayout[1].NodesCount *= (1 + ((layer0Dim[i] - (ker1Dim[i])) / stride1[i]));
			}
			layerLayout[1].NodesCount *= layerLayout[1].SubLayerCount;

			layerLayout[1].ZCount = layerLayout[1].NodesCount;
			layerLayout[1].WeightsCount = 1;
			for (unsigned i = 0; i < dim; i++)
			{
				layerLayout[1].WeightsCount *= ker1Dim[i];
			}
			layerLayout[1].WeightsCount *= layerLayout[1].SubLayerCount;

			layerLayout[1].BiasesCount = 1 * layerLayout[1].SubLayerCount;



			layerLayout[1].KerDim = ker1Dim;
			layerLayout[1].Stride = stride1;
			layerLayout[1].KerDimCount = dim;


		}

		FunctionsLayout funcLayout;
		{
			funcLayout.NeuronFunctions = new FunctionsLayout::NeuronFunction[layerLayoutCount - 1];
			{
				funcLayout.NeuronFunctions[0].f = Math::Identity;
			}
			funcLayout.NeuronFunctionsDerivatives = new FunctionsLayout::NeuronFunction[layerLayoutCount - 1];
			{
				funcLayout.NeuronFunctionsDerivatives[0].f = Math::IdentityDerivative;
			}

			funcLayout.FeedForwardCallBackFunctions = new FunctionsLayout::NetworkRelayFunction[layerLayoutCount - 1];
			{
				funcLayout.FeedForwardCallBackFunctions[0].f = LayerFunctions::ConvolutionLayerFeedForward;
			}

			funcLayout.BackPropegateCallBackFunctionsZ = new FunctionsLayout::NetworkRelayFunction[layerLayoutCount - 1];
			{
				funcLayout.BackPropegateCallBackFunctionsZ[0].f = CostFunctions::EmptyCostFunction;
			}

			funcLayout.BackPropegateCallBackFunctionsBW = new FunctionsLayout::NetworkRelayFunction[layerLayoutCount - 1];
			{
				funcLayout.BackPropegateCallBackFunctionsBW[0].f = CostFunctions::EmptyCostFunction;
			}

			funcLayout.RegularizationFunctions = new FunctionsLayout::NetworkRelayFunction[layerLayoutCount - 1];
			{
				funcLayout.RegularizationFunctions[0].f = CostFunctions::EmptyCostFunction;
			}

			funcLayout.TrainingFunctions = new FunctionsLayout::NetworkRelayFunction[layerLayoutCount - 1];
			{
				funcLayout.TrainingFunctions[0].f = CostFunctions::EmptyCostFunction;
			}
			funcLayout.CostFunction.f = CostFunctions::EmptyCostFunction;
			funcLayout.CostFunctionDerivative.f = CostFunctions::EmptyCostFunction;

		}



		NetworkPrototype n(layerLayout, funcLayout, layerLayoutCount, false);

		{


			for (unsigned i = 0; i < n.m_InputBufferCount; i++)
			{

				n.m_InputBuffer[i] = i % 3;

			}
			for (unsigned i = 0; i < n.m_WeightsCount; i++)
			{

				n.m_Weights[i] = 1;
			}

			n.FeedForward();

			pr("Input");
			pr("");
			for (unsigned i = 0; i < layer0Dim[2]; i++)
			{
				PrintMat(n.m_InputBuffer + i * layer0Dim[0] * layer0Dim[1], layer0Dim[0], layer0Dim[1]);
				pr("");
			}
			//PrintMat(n.m_InputBuffer, layer0Dim[0], layer0Dim[1]);



			pr("Result");
			pr("");
			for (unsigned i = 0; i < layer0Dim[2]; i++)
			{
				PrintMat(n.m_OutputBuffer + i * (layerLayout[1].NodesCount / (layer0Dim[2])), 1 + ((layer0Dim[0] - (ker1Dim[0])) / stride1[0]), 1 + ((layer0Dim[1] - (ker1Dim[1])) / stride1[1]));
				pr("");
}
			//PrintMat(n.m_OutputBuffer, (layer0Dim[0] - 1), (layer0Dim[1] - 1));
		}

#endif





#if setup  
		using namespace TNNT;
		Timer t;

		//DataFormating start
		t.Start();



#if MNIST

		DataSet data;
		{
			constexpr unsigned labelSize = 10;
			constexpr unsigned inputSize = 28 * 28;

			data.TrainingCount = traningCount;
			data.TrainingInputs = new float[data.TrainingCount * inputSize];
			data.TraningTargets = new float[data.TrainingCount * labelSize];

			data.ValidationCount = 10000;
			data.ValidationInputs = new float[data.ValidationCount * inputSize];
			data.ValidationTargets = new float[data.ValidationCount * labelSize];

			data.TestCount = 10000;
			data.TestInputs = new float[data.TestCount * inputSize];
			data.TestTargets = new float[data.TestCount * labelSize];

			//Data Formating start

			ProcessMNISTDataMT(10, data.TrainingInputs, data.TraningTargets, "trainLabel.idx1-ubyte", "trainIm.idx3-ubyte", data.TrainingCount);
			ProcessMNISTDataMT(10, data.ValidationInputs, data.ValidationTargets, "trainLabel.idx1-ubyte", "trainIm.idx3-ubyte", data.ValidationCount, 50000);
			ProcessMNISTDataMT(10, data.TestInputs, data.TestTargets, "testLabel.idx1-ubyte", "testIm.idx3-ubyte", data.TestCount);
			
		}
#endif 


		pr("Data formating time:" << t.Stop() << "s");




		//Prototype Network setup start


#if FullyConnected

		unsigned int layoutCount = 4;
		LayerLayout* layout = new LayerLayout[layoutCount];
		{
			layout[0].NodesCount = 28 * 28;
			layout[0].ZCount = 0;
			layout[0].BiasesCount = 0;
			layout[0].WeightsCount = 0;

			layout[1].NodesCount = 30;
			layout[1].ZCount = layout[1].NodesCount;
			layout[1].BiasesCount = layout[1].NodesCount;
			layout[1].WeightsCount = layout[1].NodesCount * layout[1 - 1].NodesCount;

			layout[2].NodesCount = 30;
			layout[2].ZCount = layout[2].NodesCount;
			layout[2].BiasesCount = layout[2].NodesCount;
			layout[2].WeightsCount = layout[2].NodesCount * layout[2 - 1].NodesCount;
			 
			layout[3].NodesCount = 10;
			layout[3].ZCount = layout[3].NodesCount;
			layout[3].BiasesCount = layout[3].NodesCount;
			layout[3].WeightsCount = layout[3].NodesCount * layout[3 - 1].NodesCount;


		}

		FunctionsLayout funcLayout;
		{

			funcLayout.NeuronFunctions = new FunctionsLayout::NeuronFunction[layoutCount - 1];
			{
				funcLayout.NeuronFunctions[0].f = Math::Sigmoid;
				funcLayout.NeuronFunctions[1].f = Math::Sigmoid;
				funcLayout.NeuronFunctions[2].f = Math::Sigmoid;
				;
			}
			funcLayout.NeuronFunctionsDerivatives = new FunctionsLayout::NeuronFunction[layoutCount - 1];
			{
				funcLayout.NeuronFunctionsDerivatives[0].f = Math::SigmoidDerivative;
				funcLayout.NeuronFunctionsDerivatives[1].f = Math::SigmoidDerivative;
				funcLayout.NeuronFunctionsDerivatives[2].f = Math::SigmoidDerivative;

			}

			funcLayout.FeedForwardCallBackFunctions = new FunctionsLayout::NetworkRelayFunction[layoutCount - 1];
			{
				funcLayout.FeedForwardCallBackFunctions[0].f = LayerFunctions::FullyConnectedFeedForward;
				funcLayout.FeedForwardCallBackFunctions[1].f = LayerFunctions::FullyConnectedFeedForward;
				funcLayout.FeedForwardCallBackFunctions[2].f = LayerFunctions::FullyConnectedFeedForward;

			}

			funcLayout.BackPropegateCallBackFunctionsZ = new FunctionsLayout::NetworkRelayFunction[layoutCount - 1];
			{
				funcLayout.BackPropegateCallBackFunctionsZ[0].f = LayerFunctions::FullyConnectedBackpropegateZ;
				funcLayout.BackPropegateCallBackFunctionsZ[1].f = LayerFunctions::FullyConnectedBackpropegateZ;
				funcLayout.BackPropegateCallBackFunctionsZ[2].f = CostFunctions::CrossEntropyDerivative;


			}

			funcLayout.BackPropegateCallBackFunctionsBW = new FunctionsLayout::NetworkRelayFunction[layoutCount - 1];
			{
				funcLayout.BackPropegateCallBackFunctionsBW[0].f = LayerFunctions::FullyConnectedBackpropegateBW;
				funcLayout.BackPropegateCallBackFunctionsBW[1].f = LayerFunctions::FullyConnectedBackpropegateBW;
				funcLayout.BackPropegateCallBackFunctionsBW[2].f = LayerFunctions::FullyConnectedBackpropegateBW;
			}



			funcLayout.CostFunction.f = CostFunctions::CrossEntropy;
			funcLayout.CostFunctionDerivative.f = CostFunctions::CrossEntropyDerivative;



			funcLayout.RegularizationFunctions = new FunctionsLayout::NetworkRelayFunction[layoutCount - 1];
			{
				funcLayout.RegularizationFunctions[0].f = RegularizationFunctions::L2Regularization;
				funcLayout.RegularizationFunctions[1].f = RegularizationFunctions::L2Regularization;
				funcLayout.RegularizationFunctions[2].f = RegularizationFunctions::L2Regularization;

			}


			funcLayout.TrainingFunctions = new FunctionsLayout::NetworkRelayFunction[layoutCount - 1];
			{
				funcLayout.TrainingFunctions[0].f = TrainingFunctions::GradientDecent;
				funcLayout.TrainingFunctions[1].f = TrainingFunctions::GradientDecent;
				funcLayout.TrainingFunctions[2].f = TrainingFunctions::GradientDecent;

			}
		}

#endif
#if Convolution

		unsigned int layoutCount = 4;
		LayerLayout* layout = new LayerLayout[layoutCount];
		{
			layout[0].NodesCount = 28 * 28;
			layout[0].ZCount = 0;
			layout[0].BiasesCount = 0;
			layout[0].WeightsCount = 0;

			layout[0].LayerDimCount = 2;
			unsigned layerDim0[2] = { 28,28 };
			layout[0].LayerDim = layerDim0;



			layout[1].NodesCount = 24*24*3;
			layout[1].ZCount = layout[1].NodesCount;
			layout[1].BiasesCount = 1*3;
			layout[1].WeightsCount = 5*5*3;

			layout[1].LayerDimCount = 2;
			layout[1].SubLayerCount = 3;

			unsigned layerDim1[2] = { 24,24};
			layout[1].LayerDim = layerDim1;

			unsigned kerDim1[2] = { 5,5 };
			layout[1].KerDim = kerDim1;
			layout[1].KerDimCount = 2;

			unsigned stride1[2] = { 1,1 };
			layout[1].Stride = stride1;



			layout[2].NodesCount = 12*12*3;
			layout[2].ZCount = layout[2].NodesCount;
			layout[2].BiasesCount = 0;
			layout[2].WeightsCount = 0;

			layout[2].LayerDimCount = 2;
			layout[2].SubLayerCount = 3;

			unsigned layerDim2[2] = { 12,12 };
			layout[2].LayerDim = layerDim2;

			layout[2].KerDimCount = 2;
			unsigned kerDim2[2] = { 2,2 };
			layout[2].KerDim = kerDim2;

			unsigned stride2[2] = { 1,1 };
			layout[2].Stride = stride2;



			layout[3].NodesCount = 10;
			layout[3].ZCount = layout[3].NodesCount;
			layout[3].BiasesCount = layout[3].NodesCount;
			layout[3].WeightsCount = layout[3].NodesCount * layout[3 - 1].NodesCount;


		}

		FunctionsLayout funcLayout;
		{

			funcLayout.NeuronFunctions = new FunctionsLayout::NeuronFunction[layoutCount - 1];
			{
				funcLayout.NeuronFunctions[0].f = Math::Sigmoid;
				funcLayout.NeuronFunctions[1].f = Math::Sigmoid;
				funcLayout.NeuronFunctions[2].f = Math::Sigmoid;
				;
			}
			funcLayout.NeuronFunctionsDerivatives = new FunctionsLayout::NeuronFunction[layoutCount - 1];
			{
				funcLayout.NeuronFunctionsDerivatives[0].f = Math::SigmoidDerivative;
				funcLayout.NeuronFunctionsDerivatives[1].f = Math::SigmoidDerivative;
				funcLayout.NeuronFunctionsDerivatives[2].f = Math::SigmoidDerivative;

			}

			funcLayout.FeedForwardCallBackFunctions = new FunctionsLayout::NetworkRelayFunction[layoutCount - 1];
			{
				funcLayout.FeedForwardCallBackFunctions[0].f = LayerFunctions::ConvolutionLayerFeedForward;
				funcLayout.FeedForwardCallBackFunctions[1].f = LayerFunctions::PoolingLayerFeedForward;
				funcLayout.FeedForwardCallBackFunctions[2].f = LayerFunctions::FullyConnectedFeedForward;

			}

			funcLayout.BackPropegateCallBackFunctionsZ = new FunctionsLayout::NetworkRelayFunction[layoutCount - 1];
			{
				funcLayout.BackPropegateCallBackFunctionsZ[0].f = LayerFunctions::PoolingLayerBackpropegateZ;
				funcLayout.BackPropegateCallBackFunctionsZ[1].f = LayerFunctions::FullyConnectedBackpropegateZ;
				funcLayout.BackPropegateCallBackFunctionsZ[2].f = CostFunctions::CrossEntropyDerivative;


			}

			funcLayout.BackPropegateCallBackFunctionsBW = new FunctionsLayout::NetworkRelayFunction[layoutCount - 1];
			{
				funcLayout.BackPropegateCallBackFunctionsBW[0].f = LayerFunctions::ConvolutionLayerBackpropegateBW;
				funcLayout.BackPropegateCallBackFunctionsBW[1].f = LayerFunctions::PoolingLayerBackpropegateBW;
				funcLayout.BackPropegateCallBackFunctionsBW[2].f = LayerFunctions::FullyConnectedBackpropegateBW;
			}



			funcLayout.CostFunction.f = CostFunctions::CrossEntropy;
			funcLayout.CostFunctionDerivative.f = CostFunctions::CrossEntropyDerivative;



			funcLayout.RegularizationFunctions = new FunctionsLayout::NetworkRelayFunction[layoutCount - 1];
			{
				funcLayout.RegularizationFunctions[0].f = RegularizationFunctions::L2Regularization;
				funcLayout.RegularizationFunctions[1].f = RegularizationFunctions::L2Regularization;
				funcLayout.RegularizationFunctions[2].f = RegularizationFunctions::L2Regularization;

			}


			funcLayout.TrainingFunctions = new FunctionsLayout::NetworkRelayFunction[layoutCount - 1];
			{
				funcLayout.TrainingFunctions[0].f = TrainingFunctions::GradientDecent;
				funcLayout.TrainingFunctions[1].f = TrainingFunctions::GradientDecent;
				funcLayout.TrainingFunctions[2].f = TrainingFunctions::GradientDecent;

			}
		}

#endif

#if test
		//Testlayout
		constexpr unsigned testlayerlayoutcount = 4;
		FCLayer l0;
		FCLayer l1;
		FCLayer l2;
		FCLayer l3;
		{
			
			l0.NodesCount = layout[0].NodesCount;
			l0.ZCount = 0;
			l0.BiasesCount = 0;
			l0.WeightsCount = 0;

			
			l1.NodesCount = layout[1].NodesCount;
			l1.ZCount = l1.NodesCount;
			l1.BiasesCount = l1.NodesCount;
			l1.WeightsCount = l1.NodesCount * l0.NodesCount;

			
			l2.NodesCount = layout[2].NodesCount;
			l2.ZCount = l2.NodesCount;
			l2.BiasesCount = l2.NodesCount;
			l2.WeightsCount = l2.NodesCount * l1.NodesCount;

			
			l3.NodesCount = layout[3].NodesCount;
			l3.ZCount = l3.NodesCount;
			l3.BiasesCount = l3.NodesCount;
			l3.WeightsCount = l3.NodesCount * l2.NodesCount;


		}
		FCLayer testlayerlayout[testlayerlayoutcount] = {l0,l1,l2,l3};



		void(*costfunction)(NetworkPrototype2*) = &CrossEntropyTest;
		void(*costfunctionderivative)(NetworkPrototype2*) = &CrossEntropyDerivativeTest;
		NetworkPrototype2 testNetwork(testlayerlayout, costfunction, costfunctionderivative,testlayerlayoutcount);
#endif

		NetworkPrototype nP(layout, funcLayout, layoutCount, true);

		//Prototype Network setup stop



		// MT prototype setup start
#if FullyConnectedMT
		unsigned int layoutCountMT = 4;
		LayerLayout* layoutMT = new LayerLayout[layoutCountMT];
		{
			layoutMT[0].NodesCount = 28 * 28;
			layoutMT[0].ZCount = 0;
			layoutMT[0].BiasesCount = 0;
			layoutMT[0].WeightsCount = 0;

			layoutMT[1].NodesCount = 30;
			layoutMT[1].ZCount = layoutMT[1].NodesCount;
			layoutMT[1].BiasesCount = layoutMT[1].NodesCount;
			layoutMT[1].WeightsCount = layoutMT[1].NodesCount * layoutMT[1-1].NodesCount;

			layoutMT[2].NodesCount = 30;
			layoutMT[2].ZCount = layoutMT[2].NodesCount;
			layoutMT[2].BiasesCount = layoutMT[2].NodesCount;
			layoutMT[2].WeightsCount = layoutMT[2].NodesCount * layoutMT[2-1].NodesCount;


			layoutMT[3].NodesCount = 10;
			layoutMT[3].ZCount = layoutMT[3].NodesCount;
			layoutMT[3].BiasesCount = layoutMT[3].NodesCount;
			layoutMT[3].WeightsCount = layoutMT[3].NodesCount * layoutMT[3-1].NodesCount;


		}

		FunctionsLayoutMT funcLayoutMT;
		{

			funcLayoutMT.NeuronFunctions = new FunctionsLayoutMT::NeuronFunction[layoutCountMT - 1];
			{
				funcLayoutMT.NeuronFunctions[0].f = Math::Sigmoid;
				funcLayoutMT.NeuronFunctions[1].f = Math::Sigmoid;
				funcLayoutMT.NeuronFunctions[2].f = Math::Sigmoid;
				;
			}
			funcLayoutMT.NeuronFunctionsDerivatives = new FunctionsLayoutMT::NeuronFunction[layoutCountMT - 1];
			{
				funcLayoutMT.NeuronFunctionsDerivatives[0].f = Math::SigmoidDerivative;
				funcLayoutMT.NeuronFunctionsDerivatives[1].f = Math::SigmoidDerivative;
				funcLayoutMT.NeuronFunctionsDerivatives[2].f = Math::SigmoidDerivative;

			}

			funcLayoutMT.FeedForwardCallBackFunctions = new FunctionsLayoutMT::NetworkRelayFunctionMT[layoutCountMT - 1];
			{
				funcLayoutMT.FeedForwardCallBackFunctions[0].f = LayerFunctionsMT::FullyConnectedFeedForward;
				funcLayoutMT.FeedForwardCallBackFunctions[1].f = LayerFunctionsMT::FullyConnectedFeedForward;
				funcLayoutMT.FeedForwardCallBackFunctions[2].f = LayerFunctionsMT::FullyConnectedFeedForward;

			}

			funcLayoutMT.BackPropegateCallBackFunctionsZ = new FunctionsLayoutMT::NetworkRelayFunctionMT[layoutCountMT - 1];
			{
				
				funcLayoutMT.BackPropegateCallBackFunctionsZ[0].f = LayerFunctionsMT::FullyConnectedBackpropegateZ;
				funcLayoutMT.BackPropegateCallBackFunctionsZ[1].f = LayerFunctionsMT::FullyConnectedBackpropegateZ;
				funcLayoutMT.BackPropegateCallBackFunctionsZ[2].f = CostFunctionsMT::CrossEntropyDerivative;

			}

			funcLayoutMT.BackPropegateCallBackFunctionsBW = new FunctionsLayoutMT::NetworkRelayFunctionMT[layoutCountMT - 1];
			{

				funcLayoutMT.BackPropegateCallBackFunctionsBW[0].f = LayerFunctionsMT::FullyConnectedBackpropegateBW;
				funcLayoutMT.BackPropegateCallBackFunctionsBW[1].f = LayerFunctionsMT::FullyConnectedBackpropegateBW;
				funcLayoutMT.BackPropegateCallBackFunctionsBW[2].f = LayerFunctionsMT::FullyConnectedBackpropegateBW;
			}



			funcLayoutMT.CostFunction.f = CostFunctionsMT::CrossEntropy;
			funcLayoutMT.CostFunctionDerivative.f = CostFunctionsMT::CrossEntropyDerivative;


			funcLayoutMT.RegularizationFunctions = new FunctionsLayoutMT::NetworkRelayFunctionMT[layoutCountMT - 1];
			{
				funcLayoutMT.RegularizationFunctions[0].f = TrainingFunctionsMT::L2Regularization;
				funcLayoutMT.RegularizationFunctions[1].f = TrainingFunctionsMT::L2Regularization;
				funcLayoutMT.RegularizationFunctions[2].f = TrainingFunctionsMT::L2Regularization;
			}
				
			funcLayoutMT.TrainingFunctions = new FunctionsLayoutMT::NetworkRelayFunctionMT[layoutCountMT - 1];

			{
				funcLayoutMT.TrainingFunctions[0].f = TrainingFunctionsMT::GradientDecent;
				funcLayoutMT.TrainingFunctions[1].f = TrainingFunctionsMT::GradientDecent;
				funcLayoutMT.TrainingFunctions[2].f = TrainingFunctionsMT::GradientDecent;
			}

		}

		NetworkPrototypeMT nMT(layoutMT, funcLayoutMT, layoutCountMT, threadN, true);
		// MT prototype setup stop

#endif


		HyperParameters params;
		{
			params.Epochs = 10;

			params.BatchCount = 10;

		}


		HyperParameters paramsMT;
		{
			paramsMT.Epochs = params.Epochs;

			paramsMT.BatchCount = params.BatchCount;

		}
#endif



		
		
#if testPerformance

#if test

		pr("Test");
		pr("Prototype2: ");
		{

			//Training start
			t.Start();
			testNetwork.SetData(&data);


#if train
			testNetwork.Train(&data, params);

			pr("Train time: " << t.Stop() << "s");
#endif
			//Traning End


			//Check Start
			t.Start();

			pr("Cost: " << testNetwork.CheckCost());
			pr("Guessrate: " << testNetwork.CheckSuccessRate());


			pr("Check time: " << t.Stop() << "s");
			//Check Stop
		}

#endif 


#if testPrototype
		pr("New");
		pr("Prototype: ");
		{

			//Training start
			t.Start();
			nP.SetData(&data);


	#if train

			

			//nP.Train(&data, params);
			
			pr("Train time: " << t.Stop() << "s");

			//.SaveParams();

			nP.LoadParams();

	#endif
			//Traning End

			
			//Check Start
			t.Start();

			pr("Cost: " << nP.CheckCost());
			pr("Guessrate: " << nP.CheckSuccessRate());

			if (nP.CheckSuccessRate() > 0.9)
			{
				nP.SaveParams();
				pr("We got it!");
			}

				

			//PArr<float>(nP.m_OutputBuffer, nP.m_OutputBufferCount);

			pr("Check time: " << t.Stop() << "s");
			//Check Stop
		}
#endif

	



#if testMultithread

		pr("MT");
		pr("Prototype: ");
		{	
#if train
			nMT.SetData(&data);
			pr("Cost: " << nMT.CheckCost());
			pr("Guessrate: " << nMT.CheckSuccessRate());

			//Training start
			t.Start();
			


		
			nMT.Train(&data, paramsMT);
	
		
			pr("Train time: " << t.Stop() << "s");
	#endif
			//Traning End
			

			t.Start();

			pr("Cost: " << nMT.CheckCost());
			pr("Guessrate: " << nMT.CheckSuccessRate());


			pr("Check time: " << t.Stop() << "s");
			//Check Stop
		}
#endif

#endif

		






		
	}

#endif




	std::cin.get();
}



