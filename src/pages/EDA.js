import React from "react";
import BreakHisTable from "../components/DataSummaryTable.js";
import DetailedBreakHisTable from "../components/DetailedDataTable.js";
import Card from "../components/Card.js";

function Home() {
  return(
    <div>
      <article class="prose mx-40 mt-10">
        <h2>Characteristics of the Dataset</h2>
        <p>Click here for a link to the original dataset and to find out more.</p>
        <p>Credits for Dataset Information (inclusive of all diagrams and tables) to [1] Spanhol, F., Oliveira, L. S., Petitjean, C., Heutte, L., A Dataset for Breast Cancer Histopathological Image Classification, IEEE Transactions on Biomedical Engineering (TBME), 63(7):1455-1462, 2016.</p>
        <blockquote class="text-xl italic font-semibold text-gray-900 dark:text-white">
            <p>The Breast Cancer Histopathological Image Classification (BreakHis) is  composed of 9,109 microscopic images of breast tumor tissue collected from 82 patients using different magnifying factors (40X, 100X, 200X, and 400X).</p>
            <p>To date, it contains 2,480  benign and 5,429 malignant samples (700X460 pixels, 3-channel RGB, 8-bit depth in each channel, PNG format).</p>
        </blockquote>
        <p>The dataset contains images of tumours belonging to two categories: benign and malignant.</p>
        <p>Benign tumours tend to grow slowly and do not spread, while malignant tumours tend to grow much faster and have the potential to invade and destroy adjacent tissues, and spread to the rest of the body.</p>
        <p>This is the structure of the BreakHis 1.0 dataset:</p>
      </article>
      <BreakHisTable />
      <article class="prose mx-40 mt-10">
        <p>There are four types of benign tumours in the dataset: adenosis (A), fibroadenoma (F), phyllodes tumor (PT), and tubular adenona (TA);  and four malignant tumors (breast cancer): carcinoma (DC), lobular carcinoma (LC), mucinous carcinoma (MC) and papillary carcinoma (PC).</p>
        <h3>Benign Tumours</h3>
      </article>
      <div class="grid grid-cols-2 gap-4 mx-40 mt-10">
        <Card title="Adenosis" description="Condition where the glands which produce milk become enlarged."/>
        <Card title="Fibroadenoma" description="Condition where the glands which produce milk become enlarged."/>
        <Card />
        <Card />
      </div>
      <article class="prose mx-40 mt-10">
        <h3>Malignant Tumours</h3>
      </article>
      <div class="grid grid-cols-2 gap-4 mx-40 mt-10">
        <Card />
        <Card />
        <Card />
        <Card />
      </div>
      <article class="prose mx-40 mt-10">
        <h3>Here is a more detailed breakdown of the BreakHist 1.0 dataset:</h3>
      </article>
      <DetailedBreakHisTable />
      <article class="prose mx-40 mt-10">
        <p>After performing EDA, the next step is to prepare the data to train and test the neural networks!</p>
      </article>
    </div>

  )
}
export default Home;