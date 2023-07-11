import React from "react";
import VGG from "../assets/images/internet_images/vgg.jpg";
import Inception from "../assets/images/internet_images/inception.PNG";
import ResNet from "../assets/images/internet_images/resnet.PNG";
import NaiveInceptionModule from "../assets/images/internet_images/naive_inception_module.PNG";
import InceptionModule from "../assets/images/internet_images/inception_module.PNG";
import FilterSizes from "../assets/images/hand_drawn/big_small_filters.jpeg";

function Plan() {
  return(
    <div>
      <article class="prose mx-40 mt-10">
        <h2>Project Plan</h2>
        <p>There is one web page dedicated to each phase of the project. This page will document the Planning phase.</p>

        <h3>Scope</h3>
        <p>The objective of this project is to fine-tune popular Convolutional Neural Network(CNN) models and determine which is the most suitable for detecting malignant tumors(cancer) in microscope scans. To familiarise myself with the theory, I read some papers and summarised what I understood, but if you want to skip straight to the implementation you could skip some of the paragraphs.</p>
        <p>These are the models I&apos;ll be using for the project:</p>
        <ul>
          <li>
            <h3>VGG19</h3>
            <p>The VGG model was first proposed in a <a href='https://arxiv.org/pdf/1409.1556.pdf' target="_blank" class="text-blue-500 hover:underline">2014 paper, "Very Deep Convolutional Neural Networks for Large-Scale Image Recognition"</a>, by computer scientists Andrew Zisserman and Karen Simonyan.</p>
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
            <p>ResNet won the ILSVRC in 2015 in image classification, detection, and localization. It was proposed in this 2016 paper by four authors.</p>
            <p>The table below shows some architectures for ImageNet. Building blocks are shown in brackets, with the numbers of blocks stacked. Downsampling is performed by conv3 1, conv4 1, and conv5 1 with a stride of 2.</p>
            <figure><img src={ResNet} /></figure>

            <h4>Motivation</h4>
            <p>The authors were driven to find a solution to two problems:</p>
            <ol>
              <li>
                <p><b>The Vanishing/Exploding Gradient Problem</b></p>
                <p>This refers to</p>
              </li>
                <p><b>The Degradation Problem</b></p>
                <p></p>
              <li>
              </li>
            </ol>
            <h4>Architecture</h4>
          </li>
          <li>
            <h3>DenseNet201</h3>
            <p>DenseNet201 is </p>
          </li>
        </ul>

        <h3>Fine-Tuning Process</h3>
        <p>I will first configure the following parameters(in this order):</p>
        <ul>
          <li>Batch Size</li>
          <li>Learning Rate</li>
          <li>Choice of Optimiser</li>
        </ul>

        <h3>Performance Metrics</h3>
        <p>These metrics will be used to evaluate model performance:</p>
        <ul>
          <li>Accuracy</li>
          <li>F1 Score</li>
          <li>ROC Curve</li>
        </ul>
      </article>
    </div>
  )
}export default Plan;