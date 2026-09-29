> ## Documentation Index
> Fetch the complete documentation index at: https://docs.edgeimpulse.com/llms.txt
> Use this file to discover all available pages before exploring further.

# Classification

> Train neural network classifiers and review model performance after feature extraction.

If you have selected the **Classification** learning block in the **Create impulse** page, a **NN Classifier** page will show up in the menu on the left. This page becomes available after you've extracted your features from your DSP block.

<Info>
  **Tutorials**

  Want to see the **Classification block** in action? Check out our tutorials:

  * [Continuous Motion Recognition](/tutorials/end-to-end/motion-recognition).
  * [Keyword spotting](/tutorials/end-to-end/keyword-spotting)
  * [Sound recognition](/tutorials/end-to-end/sound-recognition)
  * [Sensor Fusion](/tutorials/end-to-end/environmental-sensor-fusion)
</Info>

**The basic idea is that a neural network classifier will take some input data, and output a probability score that indicates how likely it is that the input data belongs to a particular class.**

So how does a neural network know what to predict? The neural network consists of several layers, each of which is made up of a number of neurons. The neurons in the first layer are connected to the neurons in the second layer, and so on. The weight of a connection between two neurons in a layer is randomly determined at the beginning of the training process. The neural network is then given a set of training data, which is a set of examples that it is supposed to predict. The network's output is compared to the correct answer and, based on the results, the weights of the connections between the neurons in the layer are adjusted. This process is repeated a number of times, until the network has learned to predict the correct answer for the training data.

A particular arrangement of layers is referred to as an architecture, and different architectures are useful for different tasks. This way, after a lot of iterations, the neural network learns; and will eventually become much better at predicting new data.

<Frame caption="NN Classifier">
  <img src="https://mintcdn.com/edgeimpulse/UMiv9oezYdq-CZ8W/.assets/images/studio-NN-Classifier-view.png?fit=max&auto=format&n=UMiv9oezYdq-CZ8W&q=85&s=d2d8fcc6ff1fc7df9b0509785fec6dba" alt="NN Classifier page with training settings and model performance panels" width="882" height="1000" data-path=".assets/images/studio-NN-Classifier-view.png" />
</Frame>

On this page, you can configure the model and the training process and, have an overview of your model performances.

## Neural Network settings

See [Neural Network Settings](/studio/projects/learning-blocks#neural-network-settings) on the Learning Block page.

## Neural Network architecture

See [Neural Network Architecture](/studio/projects/learning-blocks#neural-network-architecture) on the Learning Block page.

## Expert mode

See [Expert mode](/studio/projects/learning-blocks#expert-mode) on the Learning Block page.

## Training output

This panel displays the output logs during the training. The previous training logs can also be retrieved from the **Jobs** tab in the **Dashboard** page (enterprise feature).

## Model performances

<Frame caption="NN performances">
  <img src="https://mintcdn.com/edgeimpulse/8kB6BRmB5KYHTUny/.assets/images/NN-performances.png?fit=max&auto=format&n=8kB6BRmB5KYHTUny&q=85&s=4b42362252d18ec8a54b80834d05affc" alt="Neural network performance view with accuracy, loss, and confusion matrix results" width="1055" height="1000" data-path=".assets/images/NN-performances.png" />
</Frame>

This section gives an overview of your model performances and helps you evaluate your model. It can help you determine if the model is capable of meeting your needs or if you need to test other hyper parameters and architectures.

From the **Last training performances** you can retrieve your validation accuracy and loss.

The **Confusion matrix** is a standard tool for evaluating a model. It tabulates all of the correct and incorrect responses a model produces given a set of data. The labels on the side correspond to the actual labels in each sample, and the labels on the top correspond to the predicted labels from the model.

The **features explorer**, like in the processing block views, indicated the spatial distribution of your input features. In this page, you can visualize which ones have been correctly classified and which ones have not.

**On-device performance**: Based on the target you chose in the **Dashboard** page, we will output estimations for the *inferencing time, peak RAM usage* and *flash usage*. This will help you validate that your model will be able to run on your device based on its constraints.
