import React from "react";
import BreakHisTable from "../components/DataSummaryTable.js";
import BarGraph from "../components/DatasetBarGraphSummary.js";
import PieChartComponent from "../components/PieChartComponent.js"
import StandardDatasetSplits from '../assets/images/internet_images/standard_splits.png';

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
      <p>To increase the reliability of results, I will be using a split of 80 : 0 : 20 (no Validation set to validate the model during training) to act as a control.</p>
      <h4>Three common standard splits:</h4>
      <img src={StandardDatasetSplits} alt="logo" />
      <p>I will fine-tune each model using the 80 : 10 : 10 split. After that, I will try all above splits and use the one that results in the most optimal model performance(defined in the first and last section).</p>
      </article>
      <PieChartComponent data={controlSplitData}/>
    </div>
  )
}export default DataPrep;