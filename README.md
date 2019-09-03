# com.pneumonia.classifier

 Artificial Convolutional Neural Network For Pneuomonia Classification. This is a model generation class.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites

Java SE JDK 8,

GRADLE 5.6

### Installing (Windows)

Model file already has been generated for you in root directory - "main.bin". If you would like to retrain model you would need to parse data samples into resources folder:

1. Download and extract repository

2. Create ./data/train and ./data/test directories and paste pneumonia images there. Supported image formats are JPEG.

3. Open command line at extracted directory

4. Type:
```
gradle build
```
5. Gradle will generate jar at ../com.pneumonia.classifier/build/libs/

This will generate new model in root directory.

### Running the tests

PneumoniaClassifierTrainApp performes model evaluation right after model get trained.
Output of model accuracy is printed to console.

## Model Structure

Model consists of 6 layers:

1. Convolutional builder layer

2. Subsampling layer 

3. Convolutional builder layer

4. Subsampling layer 

5. Dense Layer 

6. Output layer

Loss function is Negative likely hood and optimization SGD. Output is driven by SOFTMAX with range 0-1.

## License

This project is licensed under the [Apache License 2](https://www.apache.org/licenses/LICENSE-2.0)
