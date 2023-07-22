import React from "react";
import BreakHisTable from "../components/DataSummaryTable.js";
import BarGraph from "../components/DatasetBarGraphSummary.js";
import PieChartComponent from "../components/PieChartComponent.js"
import StandardDatasetSplits from '../assets/images/internet_images/standard_splits-removebg.png';

function DataPrep() {
  const originalDataDistribution = [
    {
      "name": "Original Dataset Distribution",
      "Benign": 2480,
      "Malignant": 5429
    }
  ]

  const newDataDistribution = [
    {
      "name": "New Dataset Distribution",
      "Benign": 2480,
      "Malignant": 3720
    }
  ]

  const controlSplitData = [
    { name: "Group A", value: 400, fill: "#0088FE" },
    { name: "Group B", value: 300, fill: "#00C49F" },
    { name: "Group C", value: 300, fill: "#FFBB28" },
    { name: "Group D", value: 200, fill: "#FF8042" }
  ];

  return(
    <div>
      <article class="prose mx-40 mt-10"><h2>Data Preparation</h2></article>
      <BreakHisTable />
      <p class="prose mx-40 mt-10">This is the ratio of samples which are benign to those which are malignant:</p>
      <BarGraph data={originalDataDistribution} />
      <p class="prose mx-40">The dataset is moderately imbalanced and biased towards the over-represented Malignant class as the ratio of Benign : Malignant samples is 31.4 : 68.6. The dataset should be less skewed towards the Malignant samples.</p>
      <article class="prose mx-40 mt-10">

      <h3>Transforming Data</h3>
        <p>Firstly, the images will be transformed before being used to train, validate, and test the models.</p>
        <ul>
          <li>Resize each image to 224 x 224 pixels</li>
          <li>Convert each image to a Tensor</li>
        </ul>

      <h3>Undersampling the majority class</h3>
      <p>To reduce the imbalance in the dataset by increasing the ratio of Benign : Malignant samples to 40 : 60, I will use all 2480 samples from the Benign class but randomly select only 3720 samples from the Malignant class (which originally has 5429 samples).</p>
      <p>This is the new size/distribution of the dataset after undersampling the majority class (Malignant).</p>
      <div class="grid grid-cols-2 gap-10">
        <div><BarGraph data={originalDataDistribution} /></div>
        <div><BarGraph data={newDataDistribution}/></div>
      </div>

      <h3>Train, Validation and Test Split</h3>
      <h4>Control</h4>
      <p>To increase the reliability of results, I will also be using a split of 80 : 0 : 20 (no Validation set to validate the model during training) to act as a control.</p>
      <h4>Three common standard splits (image from V7 Labs)</h4>
      <img src={StandardDatasetSplits} alt="logo" />
      <p>I will try all above splits for model training/validation/testing and use the one that results in the most optimal model performance(defined in the first and last section).</p>
      <p>The training set is used to train the model for each epoch, and, simultaneously after each epoch, the validation set is used to measure model performance. This helps to prevent the model from overfitting the training data. After all epochs are completed, model performance will be confirmed using the test dataset.</p>
      <p>Usually, having more hyperparameters to tune requires a larger validation set. Using a dataset with more features and dimensions would also increase the hyperparameters of the dataset, making the model more complex.</p>
      <p>There are two important concerns to take note of when deciding the split:</p>
        <ol>
          <li>When less training data is made available to the model, there&apos;ll be more variance during the training process</li>
          <li>If less validation and testing data is made available to the model, there&apos;ll be greater variance in model performance.</li>
        </ol>
      <p>There is no single optimal split that works for all situations. The optimal split must be decided based on the selected dataset and model.</p>
      </article>
    </div>
  )
}export default DataPrep;