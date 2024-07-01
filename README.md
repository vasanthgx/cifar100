

![logo](https://github.com/vasanthgx/cifar100/blob/main/images/logo.gif)


# Project Title


**Implementation of three different methods to classify images from the CIFAR-100 dataset using CNN**
 <img src="https://github.com/Anmol-Baranwal/Cool-GIFs-For-GitHub/assets/74038190/b3fef2db-e671-4610-bb84-1d65533dc5fb" width="300" align='right'>

<br><br>


## Introduction

This project explores three methods of utilizing Convolutional Neural Networks (CNNs) to classify images in the CIFAR-100 dataset. The approaches include building a CNN from scratch, leveraging a pre-existing network, and employing transfer learning with fine-tuning. Each method is evaluated on its performance, with best practices such as early stopping, appropriate activation functions, and TensorBoard monitoring employed to optimize results. The goal is to determine the most effective technique for achieving high accuracy in image classification tasks on this complex dataset.



## Project Overview


The goal of this project is to compare different methods of building Convolutional Neural Networks (CNNs) to classify images from the CIFAR-100 dataset. The methods compared are:

1) Building a CNN from scratch.
2) Using a pre-existing network.
3) Transfer learning with fine-tuning.


## Dataset

The CIFAR-100 dataset is a collection of 60,000 color images, each 32x32 pixels, divided into 100 classes. Each class contains 600 images, with 500 for training and 100 for testing, totaling 50,000 training images and 10,000 test images. The classes are grouped into 20 superclasses, providing a more challenging classification task due to fine-grained differences between classes. Developed by the Canadian Institute For Advanced Research, CIFAR-100 is widely used for benchmarking machine learning algorithms, particularly for image recognition tasks, due to its diversity and complexity. The dataset's small image size demands efficient and powerful models for accurate classification.
![alt text](https://github.com/vasanthgx/cifar100/blob/main/images/cifar100-classes.png)

![alt text](https://github.com/vasanthgx/cifar100/blob/main/images/cifar100.png)


## Methods

1. **Building from Scratch**

Convolutional Neural Networks (CNNs) are a class of deep learning algorithms designed for processing structured grid data, such as images. They are particularly adept at capturing spatial hierarchies in data, making them the go-to choice for image recognition tasks. A typical CNN architecture consists of several layers, including convolutional layers, pooling layers, and fully connected layers.

Convolutional layers apply a series of filters to the input image, extracting features such as edges, textures, and patterns. These filters slide across the image, performing element-wise multiplication and summation, producing feature maps that highlight the presence of specific features. Pooling layers, often using max pooling, reduce the dimensionality of these feature maps while retaining essential information, making the network more computationally efficient and reducing the risk of overfitting.

Following these layers, fully connected layers integrate the extracted features to perform classification. Activation functions, such as ReLU (Rectified Linear Unit), introduce non-linearity, enabling the network to learn complex patterns. Batch normalization is often used to stabilize and accelerate training by normalizing the inputs of each layer.

CNNs have revolutionized various fields, including computer vision, medical imaging, and natural language processing, due to their ability to automatically and adaptively learn spatial hierarchies from input data. They have been instrumental in achieving state-of-the-art performance on tasks like image classification, object detection, and segmentation

2. **Using a Pre-existing Network**

The pretrained ConvNeXt Small model is an advanced Convolutional Neural Network (CNN) designed for image classification tasks. ConvNeXt represents a modern approach to CNN architecture, incorporating design principles from both traditional CNNs and the recent innovations seen in Transformer models. ConvNeXt Small is a smaller variant in the ConvNeXt family, balancing performance and computational efficiency.

Pretrained on large-scale datasets such as ImageNet, ConvNeXt Small benefits from extensive exposure to diverse image features, making it highly effective for transfer learning. By leveraging a pretrained ConvNeXt Small model, we can capitalize on its ability to extract rich and robust features from images, reducing the need for extensive training on our target dataset, such as CIFAR-100.

The architecture of ConvNeXt Small includes several stages of convolutional layers with batch normalization and activation functions like GELU. It utilizes a hierarchical structure, where the resolution of the feature maps is progressively reduced, and the depth of the network increases. This design helps in capturing both low-level and high-level features effectively.

In this project, the ConvNeXt Small model undergoes fine-tuning to adapt its learned features to the specific classes of the CIFAR-100 dataset. This involves training the model on the CIFAR-100 images while adjusting the weights of the network to improve classification performance. The result is a powerful and efficient model capable of achieving high accuracy on complex image recognition tasks.

3. **Transfer Learning + Fine Tuning**

Transfer learning and fine-tuning are powerful techniques in deep learning, particularly useful when dealing with limited datasets. Transfer learning leverages a pretrained model, which has already learned features from a large dataset, and applies it to a different but related task. This approach takes advantage of the model's ability to generalize from its previous training, reducing the need for extensive training from scratch on a new dataset.

In practice, a model like ResNet or ConvNeXt pretrained on a large-scale dataset like ImageNet can serve as a feature extractor for a smaller target dataset such as CIFAR-100. The pretrained model's lower layers, which capture general features like edges and textures, are often retained, while the upper layers, which capture more task-specific features, are modified or replaced to suit the new task.

Fine-tuning is the process of training these upper layers on the target dataset, adjusting the weights to improve performance. This often involves unfreezing some of the pretrained layers and continuing training with a lower learning rate to refine the model without destroying the learned features. Fine-tuning can significantly enhance model performance, as it adapts the generalized features to the specific nuances of the new data.

Together, transfer learning and fine-tuning accelerate the training process, improve performance, and enable the use of sophisticated models even with limited computational resources and data.


## Best Practices

**Monitoring:**

Training progress was monitored using TensorBoard, a visualization toolkit that provides insights into metrics like loss and accuracy, enabling real-time tracking and debugging. This helps in understanding the model’s performance and making necessary adjustments during training.

**Early Stopping:**

Early stopping was implemented to prevent overfitting. This technique stops the training process when the model's performance on a validation set starts to degrade, ensuring that the model does not over-learn the training data and generalizes better to new data.

**Activation Functions:**

Appropriate activation functions, such as ReLU (Rectified Linear Unit), were used to introduce non-linearity into the network. This helps the model learn complex patterns and interactions in the data, improving its ability to make accurate predictions.

**Batch Size:**

Different batch sizes were experimented with to find the optimal size for training. Batch size impacts the stability and speed of training; smaller batches provide more updates but can be noisier, while larger batches are more stable but slower.

**Data Augmentation

Data augmentation techniques were applied to improve model generalization. These techniques, such as random cropping, flipping, and rotation, artificially increase the diversity of the training dataset, helping the model become more robust to variations in input data.

**Learning Rate Scheduling:**

Learning rate scheduling was used to adjust the learning rates dynamically during training. By reducing the learning rate at predefined intervals or based on performance, the model can converge more effectively, balancing between rapid learning and fine-tuning.

**TensorBoard Visualization:**

TensorBoard is a visualization toolkit for monitoring and debugging deep learning models. It provides real-time insights into metrics like loss, accuracy, and training progress. Users can track changes, compare experiments, and understand model performance, helping to optimize and improve training processes effectively.

![alt text](https://github.com/vasanthgx/cifar100/blob/main/images/tb1.png)

## Results

![alt text](https://github.com/vasanthgx/cifar100/blob/main/images/table.png)

## Result Analysis



The results indicate varying levels of performance across different methods for classifying images from the CIFAR-100 dataset. Using a pre-existing network with pretrained weights achieved the highest accuracy at 46%, demonstrating the efficiency of leveraging learned features. Building a CNN from scratch followed with an accuracy of 36%, showcasing the potential of custom architectures tailored to specific tasks. Transfer learning combined with fine-tuning showed slightly lower accuracy at 31%, suggesting the challenge of adapting generic features to a more specific dataset context.

**Improving Results**

To enhance accuracy:

1. **Architecture Tuning:** Experiment with deeper or wider architectures to capture more intricate features.
2. **Hyperparameter Optimization:** Fine-tune learning rates, batch sizes, and regularization parameters for better model convergence.
3. **Data Augmentation:** Increase dataset diversity with techniques like rotation, flipping, and cropping to improve generalization.
4. **Ensemble Methods:** Combine predictions from multiple models for improved robustness and accuracy.
5. **Transfer Learning Variants:** Try different pretrained models or adjust the depth of fine-tuning layers for optimal feature adaptation.

By iteratively refining these aspects, it's possible to achieve higher accuracy and robustness in image classification tasks on complex datasets like CIFAR-100.











## Acknowledgements

- The CIFAR-100 dataset is provided by the [Canadian Institute For Advanced Research.](https://www.cs.toronto.edu/~kriz/cifar.html)
- [For image classification use cases, see this page for detailed examples](https://keras.io/api/applications/#usage-examples-for-image-classification-models)
- [A guide for Transfer learning and finetuning](https://keras.io/guides/transfer_learning/)






## Contact

If you have any feedback/are interested in collaborating, please reach out to me at vasanth_1627@gmail.com


## License

[![MIT License](https://img.shields.io/badge/License-MIT-green.svg)](https://choosealicense.com/licenses/mit/)

