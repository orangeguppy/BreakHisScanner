import React from "react";
import { MathJax, MathJaxContext } from "better-react-mathjax";

import VGG from "../assets/images/internet_images/vgg.jpg";
import Inception from "../assets/images/internet_images/inception.PNG";
import ResNet from "../assets/images/internet_images/resnet.PNG";
import ResNetBlock from "../assets/images/internet_images/resnet_building_block.PNG";
import NaiveInceptionModule from "../assets/images/internet_images/naive_inception_module.PNG";
import InceptionModule from "../assets/images/internet_images/inception_module.PNG";
import DenseNet from "../assets/images/internet_images/densenet.PNG";
import DenseNetSummary from "../assets/images/internet_images/densenet_summary_image.PNG"
import FilterSizes from "../assets/images/hand_drawn/big_small_filters.jpeg";
import SigmoidFxn from "../assets/images/internet_images/sigmoid.png";
import ROCSample from "../assets/images/internet_images/roc_sample.PNG";
import ConfusionMatrix from "../assets/images/internet_images/confusion_matrix.PNG";

function Plan() {
  return(
    <div>
      <article class="prose mx-40 mt-10">
        <h2>Project Plan</h2>
        <p>There is one web page dedicated to each phase of the project. This page will document the Planning phase.</p>

        <h3>Scope</h3>
        <p>The objective of this project is to fine-tune popular Convolutional Neural Network(CNN) models and determine which is the most suitable for detecting malignant tumors(cancer) in microscope scans. Because I am completely new to neural networks, to familiarise myself with the theory, I read some papers and online articles and summarised what I understood, but if you want to skip straight to the implementation you could skip some of the paragraphs.</p>
        <p>These are the models I&apos;ll be using for the project:</p>
        <p><b>*If not stated, all images depicting model architecture are from academic papers(links below) and are sadly not mine*</b></p>
        <ul>
          <li>
            <h3>VGG19</h3>
            <p>The VGG model was first proposed in a <a href='https://arxiv.org/pdf/1409.1556.pdf' target="_blank" class="text-blue-500 hover:underline">2014 paper, "Very Deep Convolutional Neural Networks for Large-Scale Image Recognition"</a>, by computer scientists Andrew Zisserman and Karen Simonyan.</p>
            <p>It won the localisation task and was the 1st runner-up in the classification task during the 2014 ILSVRC (ImageNet Large Scale Visual Recognition Competition).</p>
            <p>Below are the VGG configurations proposed by the two authors.</p>
            <figure><img src={VGG} alt="Proposed VGG configurations by the authors" /></figure>

            <h4>Motivation</h4>
            <p>This model was born from the authors&apos; attempts to improve the performance of AlexNet, a CNN model which won the Imagenet large-scale visual recognition challenge in 2012. Their findings were documented in the paper mentioned above.</p>
            <h4>Architecture</h4>
            <p>The most significant contribution of the paper, according to the authors, was evaluating the effect of increasing the depth of a neural network(pushing the number of weight layers to 16-19) while keeping the size of the convolution filters very small(3x3). These modifications can be seen in the diagram above, where the notation conv3-256 refers to a convolutional layer with 256 filters, each 3x3 in size.</p>
            <p>As the diagram below shows, a stack of three 3x3 convolutional layers has the same effective receptive field as a single 7x7 layer.</p>
            <figure><img style={{ width: 250, height: 300 }} src={FilterSizes} /></figure>

            <p>According to the paper, there are two key benefits of using a stack of smaller filters in place of a single larger one:</p>
            <ul>
              <li>
                <h4>Capacity to detect more complex features</h4>
                <p>Each layer in the VGG model uses ReLU as an activation function which introduces non-linearity. This additional non-linearity enables the model to detect more complex, non-linear mappings between input and output.</p>
                <p>The 1x1 convolutional layers in configuration C (denoted as conv1-512) carry out a linear projection onto another space with the same dimensions, further adding to the non-linearity of the model.</p>
              </li>
              <li>
                <h4>Reduced Computational Power and Time required for Training</h4>
                <p>A stack of three 3x3 convolutional filters, with C input and output channels, will have 3(3*3 * C*C) = 27C*C parameters. In contrast, a single 7x7 layer with C input and output channels will require 7*7*C*C = 49C*C parameters.</p>
                <p>Fewer parameters result in lower computational capacity, reduced memory requirements, and faster computations during forward and backward passes.</p>
              </li>
            </ul>
          </li>
          <li>
            <h3>InceptionV4</h3>
            <p>The Inception architecture was proposed in another <a href='https://arxiv.org/pdf/1409.4842v1.pdf' target="_blank" class="text-blue-500 hover:underline">2014 paper by 9 authors, called "Going Deeper with Convolutions"</a>.</p>
            <p>It won the 2014 ILSVRC (ImageNet Large Scale Visual Recognition Competition).</p>
            <p>Below shows the most successful configuration of the Inception architecture found by the authors when they wrote the paper.</p>
            <figure><img src={Inception} alt="The most successful Inception architecture when the paper was published." /></figure>

            <h4>Motivation</h4>
            <p>As mentioned by the authors, the most direct way to improve the performance of a model is to increase its size.</p>
            <p>However, this approach has two key drawbacks. Firstly, larger models are more prone to overfitting, which means that the model memorises the training data instead of detecting underlying patterns. Secondly, uniformly increasing the network size, which refers to increasing the number of layers and the number of neurons per layer by the same proportion, drastically increases computational requirements.</p>
            <p>The authors suggest that a method to resolve both issues fundamentally would be to make the architecture less dense, by promoting sparsity both between and within layers. Sparsity between layers can be encouraged by reducing the number of connections between layers, while sparsity within a layer can be reduced with various techniques like Dropout and Pruning.</p>
            <p>However, current computing infrastructure is inefficient when doing numerical calculations on non-uniform sparse data structures. The authors state that even if the number of arithmetic operations is reduced by 100 times, the overhead of lookups and cache misses for performing operations on these data structures would overshadow any increases in performance due to sparsity.</p>
            <p>Hence, the authors believe that the most practical approach would find a middle ground by using an architecture that encourages sparsity, while taking advantage of current computing infrastructure&apos;s efficiency with dense data structures.</p>
            <p>They tried to achieve this balance in Inception&apos;s architecture with the use of structures called Inception modules.</p>

            <h4>Architecture</h4>
            <p>Here are some diagrams of an Inception module from the paper.</p>
            <figure><img src={NaiveInceptionModule} /></figure>
            <figure><img src={InceptionModule} /></figure>
            <p>The structure of an Inception model is straightforward. The input is passed into four different paths(not layers). Finally, the output from all the paths is concatenated into a single output at the end of the Inception module to be passed on to the next one.</p>
            <p>In each module, there are three filters of different sizes (1x1, 3x3, and 5x5). The goal of having differently-sized filters is to capture information from the image at different levels of abstraction.</p>
            <p>The 1x1 filter does not detect spatial patterns, which refer to local variations in pixel values or intensity values in the image. It is primarily responsible for reducing the dimensionality and projections, which is why input is passed through 1x1 filters before being passed into the 3x3 and 5x5 filters. I had trouble understanding how the 1x1 filter reduced dimensionality at first, but I found a <a href='https://stats.stackexchange.com/questions/194142/what-does-1x1-convolution-mean-in-a-neural-network' target="_blank" class="text-blue-500 hover:underline">simple explanation online</a>. The reduction in dimensionality encourages sparsity, decreasing computational requirements.</p>
            <p>Smaller filters(like the 3x3) capture smaller receptive fields, allowing more localised patterns to be detected. On the other hand, larger filters(like the 5x5) are more effective at capturing more global and larger-scale patterns. In addition, because a larger filter covers a wider area, it is can also detect relationships between distant features.</p>
            <p>Max pooling divides the input feature map into non-overlapping regions, uses the highest value in the region to represent the region. This preserves the most prominent features and reduces the spatial dimension of the feature maps, further reducing computational requirements.</p>
            <p>Last but not least, by combining input from the various filters and max-pooling path, the next Inception module will be able to extract information at different levels of abstraction simultaneously from the output</p>
          </li>
          <li>
            <h3>ResNet-152</h3>
            <p>The ResNet architecture was proposed a 2016 paper, <a href='https://arxiv.org/pdf/1512.03385.pdf' target="_blank" class="text-blue-500 hover:underline">"Deep Residual Learning for Image Recognition"</a>, by four authors.</p>
            <p>ResNet won the ILSVRC in 2015 in image classification, detection, and localization.</p>
            <p>The table below shows some architectures for ImageNet. Building blocks are shown in brackets, with the numbers of blocks stacked. Downsampling is performed by conv3 1, conv4 1, and conv5 1 with a stride of 2.</p>
            <figure><img src={ResNet} /></figure>

            <h4>Motivation</h4>
            <p>The authors were driven to find a solution to two problems:</p>
            <ol>
              <li>
                <p><b>The Vanishing Gradient Problem</b></p>
                <p>Calculating the gradients for each weight involves multiplying the gradients throughout the network. Below is a diagram of the Sigmoid activation function:</p>
                <figure><img src={SigmoidFxn} /></figure>
                <p>The image above is from <a href='https://towardsdatascience.com/activation-functions-neural-networks-1cbd9f8d91d6' target="_blank" class="text-blue-500 hover:underline">here</a></p>

                <p>Activation functions like Sigmoid result in a small gradient when input values are very small or very large. When many small gradient values are multiplied to calculate the gradient with respect to a weight, the resulting gradient will also have a very small value. Because this small gradient value is used to update the weights, the changes in weight values are also very small, so the network cannot learn effectively.</p>
              </li>
              <li>
                <p><b>The Degradation Problem</b></p>
                <p>The degradation problem refers to the common situation where increasing the depth of the network decreases its performance. While the vanishing gradient and degradation problem refer to two different problems, the vanishing gradient problem significantly contributes to the degradation problem. The authors thought that adding more layers should not degrade network performance, because if a shallower network is extended by adding identity mappings, layers which don&apos;t transform the input, performance should still be the same.</p>
                <p>To implement identity mappings(the long curved arrow) in a network, the authors introduced a deep residual framework.</p>
              </li>
            </ol>

            <h4>Architecture</h4>
            <p>This is a residual building block.</p>
            <figure><img src={ResNetBlock} /></figure>
            <p>Here are some notations used in the paper and the diagram:</p>
            <p><b>x</b>: The input to a specific layer or block</p>
            <p><b>H(x)</b>: The desired mapping that maps the input to the correct output</p>
            <p><b>F(x)</b>: <b>F(x)</b> represents the mapping that is actually learned by the network. In a traditional neural network, <b>F(x) = H(x)</b></p>
            <p><b>H(x) - x</b> represents the changes that need to be applied to the input to transform it to the output. This is known as the residue.</p>
            <p>The authors hypothesised that it is easier to make the network learn to map the input to the residue, instead of the desired mapping (actual output). By setting <b>F(x)= H(x) - x</b>, we can make the neural network learn to map input to the residue, rather than the desired mapping, <b>H(x)</b>.</p>
            <p>Here is how the deep residual learning framework resolves the issues above:</p>
            <ul>
              <li>
                <p><b>Vanishing Gradient</b>: By making the network focus on learning the residues between the input and output, which are smaller in magnitude than the raw input, the gradients are more likely to remain strong.</p>
                <p><b>Degradation Problem</b>: Resolving the Vanishing Gradient problem will greatly alleviate the Degradation problem. In addition, identity mappings allow the network to skip one or more layers and directly propagate the input from an earlier layer to a later layer. The gradients can flow more easily through the network.</p>
              </li>
            </ul>
          </li>
          <li>
            <h3>DenseNet201</h3>
            <p>DenseNet was proposed in a <a href='https://arxiv.org/pdf/1608.06993.pdf' target="_blank" class="text-blue-500 hover:underline"> 2017 paper titled "Densely Connected Convolutional Networks"</a> by four authors.</p>
            <p>It won the Best Paper Award in the 2017 Computer Vision and Pattern Recognition competition 2017, and has accumulated over 2000 citations to this day. It was produced by a collaboration between Cornwell University, TsingHua University and Facebook AI Research (FAIR).</p>
            <figure><img src={DenseNet} /></figure>

            <h4>Motivation</h4>
            <p>Similar to ResNet, the creators of DenseNet wanted to tackle the Vanishing Gradient problem.</p>
            <p>State-of-the-art solutions at the time alleviated the problem by using short paths from earlier layers to later layers, such as ResNet, Highway Networks, and FractalNets.</p>
            <p>They wanted to embody this design principle with a simple connectivity pattern that maximised information flow between layers of the network.</p>

            <h4>Architecture</h4>
            <p>All layers are directly connected with each other. Each layer obtains inputs from all preceding layers, and passes all preceding input, together with its own input, to all subsequent layers.</p>
            <figure><img src={DenseNetSummary} /></figure>
            <p>While a traditional convolutional network with L layers would have L connections (one connection between consecutive layers), for DenseNet it would be L(L+1) / 2 direct connections, since for each layer, all preceding layers&apos; feature maps are fed as inputs, and the layer&apos;s output is fed into all subsequent layers.</p>
            <p>The authors highlighted that while ResNet combines features with summation before passing them into a layer, DenseNet concatenates features to combine them.</p>
          </li>
        </ul>

        <h3>Fine-Tuning and Training Process</h3>
        <p>I will try all parameter combinations and train/validate/test dataset splits (using values stated below) and select the one that produces the highest F1 Score, using Area under the ROC Curve as a tie-breaker. I want to try a greater range of values, but due to time constraints I will only test with these values in the meantime.</p>
        <ul>
          <li>
            <b>Batch Size</b>: 32, 64, 128
            <p>A batch is a subset of the training data. Instead of updating the model&apos;s parameters based on the entire training dataset in a single iteration, the training data is divided into smaller batches, and the model parameters are updated based on the gradients computed after each batch is processed.</p>
            <p>Smaller batch sizes might prevent the model from getting stuck in a poor local minima. This is because the model will encounter more diverse examples which can facilitate a more comprehensive exploration of the optimization landscape. However, smaller batch sizes may destabilise the training process and make parameter updates noisier and fluctuate more. Smaller sizes might introduce more variation since each batch can only provide a partial estimate of the gradient. Thus, smaller batches may need more iterations for convergence.</p>
            <p>On the other hand, larger batch sizes might use computational resources more efficiently. Larger batch sizes may also lead to better generalisation because having more samples per batch can enable the model to detect larger-scale patterns and trends.</p>
          </li>
          <li><b>Number of Epochs</b>: 11
            <p>I&apos;ll be using pre-trained models so that I can keep the number of epochs small.</p>
            <p>Pre-trained models are usually trained on huge, diverse datasets to tackle tasks like image classification, so they often already learned many general features from these tasks.</p>
          </li>

          <li>
            <b>Learning Rate</b>: 0.01, 0.001
            <p>This refers to the rate at which the parameter values change during each iteration of an optimisation algorithm. Smaller learning rates may mean that a model takes a longer time to converge, and vice versa.</p>
            <p>Smaller learning rates often prevent the model from overshooting the optimal solution, and explore the local minima more thoroughly.</p>
            <p>Larger learning rates, conversely, often help models to escape poor local minima but may increase the risk of overshooting the optimal solution.</p>
          </li>

          <li>
            <b>Choice of Optimiser</b>: SGD(Stochastic Gradient Descent), Adam, Lion
            <p>Optimisers are algorithms used to update model parameters to minimise loss.</p>

          </li>
          <li>
            <b>Weight Decay</b>: 0, 0.1, 0.001, 0.0001
            <p>Weight Decay is a regularisation technique which aims to reduce overfitting so that models can generalise better and perform better with new data not used for training.</p>
            <p>To prevent overfitting, the general approach is to prevent the model from becoming too complex. One way would be to reduce the number of parameters, but this is not ideal because having more parameters facilitates relationships between different parts of the model.</p>
            <p>The goal of introducing Weight Decay is to allow the model to have as many parameters as required, but at the same time prevent the model from becoming too complex(by preventing model weights from getting too large) to prevent overfitting.</p>
          </li>
          <li><b>Dropout Rate</b>:
            <ul>
              <li><b>No Dropout</b></li>
              <li><b>Symmetric Dropout (all layers have the same Dropout)</b>: 0.1</li>
              <p>Like Weight Decay, Dropout is a regularisation technique that also aims to reduce overfitting. Dropout aims to mitigate the problem by randomly deactivating some neurons during the training process.</p>
              <p>The key idea is to prevent the network from becoming overly reliant on specific neurons during training. During each training iteration, some fraction (specified by the Dropout rate) of neurons in a layer has their output set to 0. The weights are not modified during Dropout, only neuron activations are affected.</p>
            </ul>
          </li>
        </ul>
        <h3>Performance Metrics</h3>
        <p>These metrics will be used to visualise and evaluate model performance.</p>
        <p><b>T</b>=True, <b>F</b>=False, <b>P</b>=Positive, <b>N</b>=Negative</p>
        <ul>
          <li>
            <b>Accuracy</b> = <MathJax inline dynamic>{"\\(\\frac{TP + TN}{TP + TN + FP + FN} \\)"}</MathJax>
            <p>Accuracy measures the proportion of correctly classified samples out of the all samples in the dataset.</p>
          </li>

          <li>
            <b>Confusion Matrix</b>
            <p>The Confusion Matrix provides a comprehensive view of the model's predictions by comparing them with the actual ground truth labels.</p>
            <p>Here is the format of a Confusion Matrix, the image is from <a href='https://towardsdatascience.com/understanding-confusion-matrix-a9ad42dcfd62' target="_blank" class="text-blue-500 hover:underline">here</a></p>
            <figure><img src={ConfusionMatrix} /></figure>
          </li>

          <li>
            <b>F1 Score</b> = <MathJax inline dynamic>{"\\(\\frac{precision * recall}{precision + recall} \\)"}</MathJax> = <MathJax inline dynamic>{"\\(\\frac{TP}{TP + 1/2(FP + FN)} \\)"}</MathJax>
            <br />
            Precision = <MathJax inline dynamic>{"\\(\\frac{TP}{TP + FP} \\)"}</MathJax>
            <br />
            Recall = <MathJax inline dynamic>{"\\(\\frac{TP}{TP + FN} \\)"}</MathJax>
            <p>F1 Score is commonly used as a performance metric for binary classification problems. It is the harmonic mean(shown above) of precision and recall, and takes values from 0 to 1. A higher F1 score indicates a better balance between precision and recall, as it shows that the model makes accurate positive predictions while minimising false positives and false negatives.</p>
          </li>
          <li><b>AUC - ROC Curve (Area Under Receiver Operating Characteristic Curve)</b>
            <p>The ROC Curve is a graph that shows how the performance of a classification model varies across all classification thresholds.</p>
            <b>True Positive Rate (TPR) / Recall</b> = <MathJax inline dynamic>{"\\(\\frac{TP}{TP + FN} \\)"}</MathJax>
            <br />
            <b>False Positive Rate (FPR)</b> = <MathJax inline dynamic>{"\\(\\frac{FP}{FP + TN} \\)"}</MathJax>
            <p>Here is an example of an ROC Curve from a Machine Learning course by Google:</p>
            <figure><img src={ROCSample} /></figure>
            <p>If the classification threshold is lowered, more samples will be categorised as positive, increasing both True Positives and False Positives. This is why the curve increases as classification threshold increases.</p>
            <p>The Area under the ROC Curve considers model performance at all classification thresholds, and provides a single value which represents the models ability to distinguish between classes, regardless of the classification threshold.</p>
          </li>
        </ul>
        <h3>Managing the Machine Learning Lifecycle</h3>
          <p>I will Fine-Tune models using PyTorch.</p>
          <p>I&apos;ll use MLFlow (implementation documented in a following section) to manage the machine learning lifecycle, from initial model development to deployment and more.</p>

        <h3>References</h3>
          <p>All links referenced in the text</p>
          <a href="https://cs230.stanford.edu/files/Notation.pdf" target="_blank" class="text-blue-500 hover:underline">https://cs230.stanford.edu/files/Notation.pdf</a>
          <p></p>
          <a href="https://towardsdatascience.com/convolutional-neural-network-for-breast-cancer-classification-52f1213dcc9" target="_blank" class="text-blue-500 hover:underline">https://towardsdatascience.com/convolutional-neural-network-for-breast-cancer-classification-52f1213dcc9</a>
          <p></p>
          <a href="https://towardsdatascience.com/review-densenet-image-classification-b6631a8ef803" target="_blank" class="text-blue-500 hover:underline">https://towardsdatascience.com/review-densenet-image-classification-b6631a8ef803</a>
          <p></p>
          <a href="https://towardsdatascience.com/the-w3h-of-alexnet-vggnet-resnet-and-inception-7baaaecccc96" target="_blank" class="text-blue-500 hover:underline">https://towardsdatascience.com/the-w3h-of-alexnet-vggnet-resnet-and-inception-7baaaecccc96</a>
          <p></p>
          <a href="https://towardsdatascience.com/understanding-and-calculating-the-number-of-parameters-in-convolution-neural-networks-cnns-fc88790d530d" target="_blank" class="text-blue-500 hover:underline">https://towardsdatascience.com/understanding-and-calculating-the-number-of-parameters-in-convolution-neural-networks-cnns-fc88790d530d</a>
          <p></p>
          <a href="https://towardsdatascience.com/classification-models-and-thresholds-97821aa5760f" target="_blank" class="text-blue-500 hover:underline">https://towardsdatascience.com/classification-models-and-thresholds-97821aa5760f</a>
          <p></p>
          <a href="https://towardsdatascience.com/this-thing-called-weight-decay-a7cd4bcfccab" target="_blank" class="text-blue-500 hover:underline">https://towardsdatascience.com/this-thing-called-weight-decay-a7cd4bcfccab</a>
          <p></p>
          <a href="https://developers.google.com/machine-learning/crash-course/classification/roc-and-auc#:~:text=An%20ROC%20curve%20(receiver%20operating,model%20at%20all%20classification%20thresholds." target="_blank" class="text-blue-500 hover:underline">https://developers.google.com/machine-learning/crash-course/classification/roc-and-auc#:~:text=An%20ROC%20curve%20(receiver%20operating,model%20at%20all%20classification%20thresholds.</a>
      </article>
    </div>
  )
}export default Plan;