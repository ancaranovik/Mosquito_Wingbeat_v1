> ## Documentation Index
> Fetch the complete documentation index at: https://docs.edgeimpulse.com/llms.txt
> Use this file to discover all available pages before exploring further.

# Learning blocks

> Train models with built-in learning blocks, configure neural network settings, and access expert mode.

After extracting meaningful features from the raw signal using signal processing, you can now train your model using a learning block. We provide several pre-defined learning blocks:

* [Classification (Keras)](/studio/projects/learning-blocks/blocks/classification)
* [Regression (Keras)](/studio/projects/learning-blocks/blocks/regression)
* [Anomaly Detection (K-means)](/studio/projects/learning-blocks/blocks/anomaly-detection-k-means)
* [Anomaly Detection (GMM)](/studio/projects/learning-blocks/blocks/anomaly-detection-gmm)
* [Visual anomaly detection (FOMO-AD)](/studio/projects/learning-blocks/blocks/visual-anomaly-detection-fomo-ad)
* [Image Classification (using Transfer Learning)](/studio/projects/learning-blocks/blocks/transfer-learning-images)
* [Keyword Spotting (using Transfer Learning)](/studio/projects/learning-blocks/blocks/transfer-learning-keyword)
* [Object Detection (using MobileNetV2 SSD FPN)](/studio/projects/learning-blocks/blocks/object-detection)
* [Object Detection (using FOMO)](/studio/projects/learning-blocks/blocks/object-detection/fomo)
* [Classical ML](/studio/projects/learning-blocks/blocks/classical-ml)

Miss an architecture? You can [create a custom learning block](/studio/organizations/custom-blocks/custom-learning-blocks), with PyTorch, Keras or scikit-learn to bring your custom architecture and train it with Edge Impulse.
If you already have a trained model, you can also [Bring Your Own Model (BYOM)](/studio/projects/dashboard/byom) to directly profile it and deploy it on your edge devices.

If you are familiar with TensorFlow and Keras, in most blocks, you can use the **[Switch to Expert mode](/studio/projects/learning-blocks/expert-mode)** button to access the full Keras API for custom architectures, [rebalance your weights](/knowledge/guides/increasing-model-performance#class-imbalance), change the [optimizer](/knowledge/concepts/machine-learning/neural-networks/optimizers), and more.

For most of the learning blocks provided by Edge Impulse (Keras and Transfer Learning-based blocks), a view similar to the one below is available. See the dedicated learning block page for specific details when it differs.

<Frame caption="Classifier view">
  <img src="https://mintcdn.com/edgeimpulse/UMiv9oezYdq-CZ8W/.assets/images/studio-classifier-view.png?fit=max&auto=format&n=UMiv9oezYdq-CZ8W&q=85&s=6bef505572349b8f47e8829634bd6bba" alt="Classifier page with model settings, training controls, and performance results" width="1113" height="1000" data-path=".assets/images/studio-classifier-view.png" />
</Frame>

## Neural Network settings

<Frame caption="NN Settings">
  <img src="https://mintcdn.com/edgeimpulse/UMiv9oezYdq-CZ8W/.assets/images/studio-NN-settings.png?fit=max&auto=format&n=UMiv9oezYdq-CZ8W&q=85&s=a9e474d03ca16ef8d6c3946a0a0c9850" alt="Neural network settings panel with training parameters for a learning block" width="938" height="1000" data-path=".assets/images/studio-NN-settings.png" />
</Frame>

* **Number of training cycles:** Each time the training algorithm makes one complete pass through all of the training data with back-propagation and updates the model's parameters as it goes, it is known as an epoch or training cycle.
* **Use Learned Optimizer (VeLO):** Use a neural network as an optimizer to calculate gradients and learning rates. For optimal results with VeLO, it is recommended to use as large a batch size as possible, potentially equal to the dataset's size. See [Learned Optimizer (VeLO)](/knowledge/concepts/machine-learning/neural-networks/learned-optimizer-velo)
* **Learning rate:** The learning rate controls how much the model's internal parameters are updated during each step of the training process. Or, you can also see it as how fast the neural network will learn. If the network overfits quickly, you can reduce the learning rate.
* **Training processor:** We recommend to choose the CPU training processor for projects with small datasets. GPU training processors are available as well for larger training jobs.
* **Validation set size:** The percentage of your training set held apart for validation, a good default is 20%.
* **Split train/validation set on metadata key:** Prevent group data leakage between train and validation datasets using sample metadata. Given a metadata key, samples with the same value for that key will always be on the same side of the validation split. Leave empty to disable. Also, see [metadata](/studio/projects/data-acquisition/dataset/metadata).
* **Batch size:** The batch size used during training. If not set, we'll use the default value. Training may fail if the batch size is too high.
* **Auto-weight classes** While training, pay more attention to samples from under-represented classes. Might help make the model more robust against overfitting if you have little data for some classes.
* **Profile int8 model:** Profiling the quantized model might take a long time on large datasets. Disable this option to skip profiling.

### Data augmentation settings

Edge Impulse provides easy to use data augmentation options for several combinations of data and model types. If available, these settings will appear within the Neural Network settings. Augmentation can be applied by enabling the corresponding checkbox and entering associated parameters if applicable.

#### Audio classification

Augmentation is applied to spectrograms, such as those generated by the MFCC and MFE processing blocks, rather than the raw audio signal. The augmentation options include masking time bands, masking frequency bands, and warping the time axis using the [SpecAugment](https://arxiv.org/abs/1904.08779) method, in addition to adding Gaussian noise.

<Frame caption="Data augmentation options for an audio classification model.">
  <img src="https://mintcdn.com/edgeimpulse/VpF_Einsy6qHCvhZ/.assets/images/data-augmentation-audio.png?fit=max&auto=format&n=VpF_Einsy6qHCvhZ&q=85&s=7cad56b7e840cdfe8b45ba07bddae87d" alt="Audio data augmentation settings with noise and pitch adjustment options" width="645" height="306" data-path=".assets/images/data-augmentation-audio.png" />
</Frame>

#### Image classification

When using a transfer learning block in your impulse for image classification tasks, augmentation can be enabled through a single checkbox. When enabled, random transformations are applied to your original images that include flipping (horizontally), zooming, cropping, and varying brightness.

<Frame caption="Data augmentation options for an image classification transfer learning model.">
  <img src="https://mintcdn.com/edgeimpulse/VpF_Einsy6qHCvhZ/.assets/images/data-augmentation-image-classification.png?fit=max&auto=format&n=VpF_Einsy6qHCvhZ&q=85&s=a0338d21b4658094f6a274c0c13d99db" alt="Image classification augmentation settings with crop, rotation, and brightness options" width="684" height="153" data-path=".assets/images/data-augmentation-image-classification.png" />
</Frame>

#### Object detection

The choice of augmentation options will depend on the object detection model that has been selected. For example, a FOMO model will offer a single checkbox that enables flipping (horizontally), zooming, cropping, rotating, varying brightness, and varying contrast; other models will offer different parameters based on the model head selected; and some other model types may not offer augmentation.

## Neural Network architecture

For classification and regression tasks, you can edit the layers directly from the web interface. Depending on your project type, we may offer to choose between different architecture presets to help you get started.

<Frame caption="NN Architecture">
  <img src="https://mintcdn.com/edgeimpulse/8kB6BRmB5KYHTUny/.assets/images/NN-architecture.png?fit=max&auto=format&n=8kB6BRmB5KYHTUny&q=85&s=7bb4b0909f5db303ca027b978048858c" alt="Neural network architecture editor with layer configuration controls" width="1277" height="1000" data-path=".assets/images/NN-architecture.png" />
</Frame>

The neural network architecture takes as inputs your extracted features, and pass the features to each layer of your architecture. In the classification case, the last used layer is a **softmax layer**. It is this last layer that gives the probability of belonging to one of the classes.

From the *visual (simple) mode*, you can add the following layers:

<Frame caption="NN layers">
  <img src="https://mintcdn.com/edgeimpulse/8kB6BRmB5KYHTUny/.assets/images/NN-layers.png?fit=max&auto=format&n=8kB6BRmB5KYHTUny&q=85&s=4140e121717d69d60202a736ff7a42b6" alt="Neural network layer list showing editable dense and dropout layers" width="905" height="1000" data-path=".assets/images/NN-layers.png" />
</Frame>

For Transfer Learning tasks like Audio or Image Transfer learning and Object Detection, you can select which pre-trained model is more suited for your use case and edit the last layers parameters to be trained:

<Frame caption="MobileNetV1 96x96 0.25 pre-trained model">
  <img src="https://mintcdn.com/edgeimpulse/x9Ga-7v4NxdQ7jXX/.assets/images/image-transfer-learning-nn-architecture.png?fit=max&auto=format&n=x9Ga-7v4NxdQ7jXX&q=85&s=4e58e57f8900dff2d58a7f8bf32802b6" alt="MobileNetV1 transfer learning architecture settings for image classification" width="1164" height="766" data-path=".assets/images/image-transfer-learning-nn-architecture.png" />
</Frame>

## Expert mode

If have advanced knowledge in machine learning and Keras, you can switch to the *Expert Mode* and access the full Keras API to use custom architectures:

<Frame caption="Switch to expert mode">
  <img src="https://mintcdn.com/edgeimpulse/bl3dk7kPQkfGIkjp/.assets/images/switch-to-expert-mode.png?fit=max&auto=format&n=bl3dk7kPQkfGIkjp&q=85&s=680616322dfe9397f6a4c25725df070e" alt="Learning block page with the Switch to expert mode button" width="1116" height="304" data-path=".assets/images/switch-to-expert-mode.png" />
</Frame>

<br />

<Frame caption="NN expert mode">
  <img src="https://mintcdn.com/edgeimpulse/8kB6BRmB5KYHTUny/.assets/images/NN-expert-mode.png?fit=max&auto=format&n=8kB6BRmB5KYHTUny&q=85&s=a33aa1802039c835e12c9403fc649348" alt="Expert mode editor showing Keras code for a neural network architecture" width="1175" height="1000" data-path=".assets/images/NN-expert-mode.png" />
</Frame>

You can use the expert mode to change your [loss function](/knowledge/concepts/machine-learning/neural-networks/loss-functions), [optimizer](/knowledge/concepts/machine-learning/neural-networks/optimizers), print your model architecture and even set an early stopping callback to prevent overfitting your model.

Also see the dedicated page about the [expert mode](/studio/projects/learning-blocks/expert-mode)
