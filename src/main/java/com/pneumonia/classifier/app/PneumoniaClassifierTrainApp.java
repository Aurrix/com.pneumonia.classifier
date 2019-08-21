/*
 * Copyright 2002-2018 the original author or authors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      https://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package com.pneumonia.classifier.app;

import java.io.File;
import java.io.IOException;
import java.util.Random;

import org.apache.log4j.BasicConfigurator;
import org.datavec.api.io.labels.ParentPathLabelGenerator;
import org.datavec.api.split.FileSplit;
import org.datavec.image.recordreader.ImageRecordReader;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.SubsamplingLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.iterator.MultipleEpochsIterator;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * @author Alisher Urunov
 *
 */
public class PneumoniaClassifierTrainApp {

	/**
	 * @param args
	 */
	
	public static Logger logger = LoggerFactory.getLogger(PneumoniaClassifierTrainApp.class);
	
	public static void main(String[] args) throws IOException {
		BasicConfigurator.configure();
		int height = 20;
		int width = 20;
		int nChannels = 1;
		int batchSize = 20;
		int output = 2;
		int nEpochs = 20;
		int iterations = 1;
		int seed = 123;
		Random randomGenerator = new Random(seed);
		
		double learningRate = 0.01;
		int hiddenNodes = 20;
		boolean save = true;
		logger.info("Loading data...");
		
		File trainData = new File ("./data/train/");
		File testnData = new File ("./data/test/");
		
		FileSplit trainSplit = new FileSplit(trainData, new String[] {"jpeg"}, randomGenerator);
		FileSplit testSplit = new FileSplit(testnData, new String[] {"jpeg"}, randomGenerator);
	
		ParentPathLabelGenerator labelGenerator = new ParentPathLabelGenerator();
		ImageRecordReader recordReader = new ImageRecordReader(height, width, nChannels,labelGenerator);
		ImageRecordReader testRecordReader = new ImageRecordReader(height, width, nChannels,labelGenerator);
				
		logger.info("Preparing dataset");
        ImagePreProcessingScaler preProcessor = new ImagePreProcessingScaler(0, 1);
		DataSetIterator iterator = new RecordReaderDataSetIterator(recordReader, batchSize, 1, output);
		DataSetIterator testIterator = new RecordReaderDataSetIterator(testRecordReader, batchSize, 1, output);

		///////////////////////////////////////////////////////////////
		
		MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
				.seed(seed)
				.iterations(iterations) // Training iterations as above
				.regularization(true).l2(0.0005)
				/*
				 Uncomment the following for learning decay and bias
				 */
				.learningRate(.01)//.biasLearningRate(0.02)
				//.learningRateDecayPolicy(LearningRatePolicy.Inverse)
				 //.lrPolicyDecayRate(0.001).lrPolicyPower(0.75)
				.weightInit(WeightInit.XAVIER)
				.optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
				.updater(Updater.NESTEROVS).momentum(0.9)
				.list()
				.layer(0, new ConvolutionLayer.Builder(5, 5)
				//nIn and nOut specify depth. nIn here is the nChannels and
				 //nOut is the number of filters to be applied
				.nIn(nChannels)
				.stride(1, 1)
				.nOut(20)
				.activation(Activation.IDENTITY)
				.build())
				.layer(1, new SubsamplingLayer
				 .Builder(SubsamplingLayer.PoolingType.MAX)
				.kernelSize(2,2)
				.stride(2,2)
				.build())
				.layer(2, new ConvolutionLayer.Builder(5, 5)
				.stride(1, 1)
				.nOut(50)
				.activation(Activation.IDENTITY)
				.build())
				.layer(3, new SubsamplingLayer.Builder(SubsamplingLayer
				 .PoolingType.MAX)
				.kernelSize(2,2)
				.stride(2,2)
				.build())
				.layer(4, new DenseLayer.Builder().activation(Activation.RELU)
				.nOut(500).build())
				.layer(5, new OutputLayer
				 .Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
				.nOut(output)
				.activation(Activation.SOFTMAX)
				.build())
				.setInputType(InputType.convolutionalFlat(height,width,1)) 
				.backprop(true).pretrain(false).build();

		
		MultiLayerNetwork model = new MultiLayerNetwork(conf);
		model.init();
	    model.setListeners(new ScoreIterationListener(50));
		
		 // Visualizing Network Training
//        UIServer uiServer = UIServer.getInstance();
//        StatsStorage statsStorage = new InMemoryStatsStorage();
//        uiServer.attach(statsStorage);
//        model.setListeners((IterationListener) new StatsListener( statsStorage),new ScoreIterationListener(iterations));

        // Load data
			recordReader.initialize(trainSplit);
	        preProcessor.fit(iterator);
	        iterator.setPreProcessor(preProcessor);
	        MultipleEpochsIterator trainIter;
	        trainIter = new MultipleEpochsIterator(nEpochs, iterator);
	        model.fit(trainIter);

			
			
			
			logger.info("Evaluate model....");
			testRecordReader.initialize(testSplit);
	        preProcessor.fit(testIterator);
	        testIterator.setPreProcessor(preProcessor);
	        Evaluation eval = model.evaluate(testIterator);
	        logger.info(eval.stats(true));
			
			logger.info("****************Test finished********************");

			
	        
	        if (save) {
	            logger.info("Save model....");
	            ModelSerializer.writeModel(model,  "model.bin", true);
	        }
	}
	
}
