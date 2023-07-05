import React from "react";
import VGG from "../assets/images/internet_images/vgg.jpg"
import FilterSizes from "../assets/images/hand_drawn/big_small_filters.jpeg"
function Plan() {
  return(
    <div>
      <article class="prose mx-40 mt-10">
        <h2>Project Plan</h2>
        <p>There is one web page dedicated to each phase of the project. This page will document the Planning phase.</p>

        <h3>Scope</h3>
        <p>The objective of this project is to fine-tune popular Convolutional Neural Network(CNN) models and determine which is the most suitable for detecting malignant tumors(cancer) in microscope scans.</p>
        <p>These are the models I&apos;ll be using for the project:</p>
        <ul>
          <li>
            <h3>VGG19</h3>
            <p>The VGG model was first proposed in a <a href='https://arxiv.org/pdf/1409.1556.pdf' target="_blank" class="text-blue-500 hover:underline">2014 paper, "Very Deep Convolutional Neural Networks for Large-Scale Image Recognition"</a>, by computer scientists Andrew Zisserman and Karen Simonyan.</p>
            <p>Below are the VGG configurations proposed by the two authors.</p>
            <figure><img src={VGG} alt="Proposed VGG configurations by the authors" /></figure>

            <h4>Significance</h4>
            <p>This model was born from the authors&apos; attempts to improve the performance of AlexNet, a CNN model which won the Imagenet large-scale visual recognition challenge in 2012. These findings were documented in the paper mentioned above.</p>
            <p>The most significant contribution of the paper, according to the authors, was the evaluation of the effect of increasing the depth of a neural network(pushing the number of weight layers to 16-19) while keeping the size of the convolution filters very small(3x3). These modifications can be seen in the diagram extracted from the paper, where the notation conv3-256 refers to a convolutional layer with 256 filters, each 3x3 in size.</p>
            <p>As the diagram below shows, a stack of three 3x3 convolutional layers has the same effective receptive field as a single 7x7 layer.</p>
            <figure><img style={{ width: 250, height: 300 }} src={FilterSizes} /></figure>

            <p>According to the paper, there are two key benefits of using a stack of smaller filters in place of a single larger one:</p>
            <ul>
              <li>
                <h4>Capacity to detect more complex features</h4>
                <p></p>
              </li>
              <li>
                <h4>Reduced Computational Power and Training Time required</h4>
                <p></p>
              </li>
            </ul>
          </li>
          <li>
            <h3>ResNet-152</h3>
            <p>ResNet-152 is</p>
          </li>
          <li>
            <h3>DenseNet201</h3>
            <p>DenseNet201 is </p>
          </li>
          <li>
            <h3>InceptionV4</h3>
            <p>InceptionV4 is </p>
          </li>
        </ul>

        <h3>Performance Metrics</h3>
        <p>These metrics will be used to evaluate model performance:</p>
        <ul>
          <li>Accuracy</li>
          <li>F1 Score</li>
          <li>ROC Curve</li>
        </ul>

        <h3>References</h3>
        <ul>
          <li class="list-none">
            <a href="https://arxiv.org/pdf/1409.1556.pdf" target="_blank" class="text-blue-500 hover:underline">Very Deep Convolutional Networks for Large-Scale Image Recognition by Karen Simonyan, Andrew Zisserman</a>
          </li>

          <li class="list-none">
            <a href="https://arxiv.org/pdf/1409.1556.pdf" target="_blank" class="text-blue-500 hover:underline">Very Deep Convolutional Networks for Large-Scale Image Recognition by Karen Simonyan, Andrew Zisserman</a>
          </li>
        </ul>
      </article>
    </div>
  )
}export default Plan;