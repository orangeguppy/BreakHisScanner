import React from "react";
import BreakHisTable from "../components/DataSummaryTable.js";
import BarGraph from "../components/DatasetBarGraphSummary.js";
import DetailedBreakHisTable from "../components/DetailedDataTable.js";
import Card from "../components/Card.js";

import Adenosis from '../assets/images/tumour_samples/benign/adenosis_100x.png';
import Fibroadenoma from '../assets/images/tumour_samples/benign/fibroadenoma_100x.png';
import Phyllodes from '../assets/images/tumour_samples/benign/phyllodes_100x.png';
import TubularAdenoma from '../assets/images/tumour_samples/benign/tubular_adenoma_100x.png';

import DuctalCarcinoma from "../assets/images/tumour_samples/malignant/ductal_carcinoma_100x.png";
import LobularCarcinoma from "../assets/images/tumour_samples/malignant/lobular_carcinoma_100x.png";
import MucinousCarcinoma from "../assets/images/tumour_samples/malignant/mucinous_carcinoma_100x.png";
import PapillaryCarcinoma from "../assets/images/tumour_samples/malignant/papillary_carcinoma_100x.png";

function EDA() {
  const originalDataDistribution = [
    {
      "name": "Original Dataset Distribution",
      "Benign": 2480,
      "Malignant": 5429
    }
  ]

  return(
    <div>
      <article class="prose mx-40 mt-10">
        <h2>Exploratory Data Analysis</h2>
        <a href="https://web.inf.ufpr.br/vri/databases/breast-cancer-histopathological-database-breakhis/" target="_blank" class="text-blue-500 hover:underline">Click here for a link to the original dataset and to find out more.</a>
        <p>Credits for Dataset Information (inclusive of all diagrams and tables) to [1] Spanhol, F., Oliveira, L. S., Petitjean, C., Heutte, L., A Dataset for Breast Cancer Histopathological Image Classification, IEEE Transactions on Biomedical Engineering (TBME), 63(7):1455-1462, 2016.</p>
        <blockquote class="text-xl italic font-semibold text-gray-900 dark:text-white">
            <p>The Breast Cancer Histopathological Image Classification (BreakHis) is  composed of 9,109 microscopic images of breast tumor tissue collected from 82 patients using different magnifying factors (40X, 100X, 200X, and 400X).</p>
            <p>To date, it contains 2,480  benign and 5,429 malignant samples (700X460 pixels, 3-channel RGB, 8-bit depth in each channel, PNG format).</p>
        </blockquote>
        <p>The dataset contains images of tumours belonging to two categories: benign and malignant.</p>
        <p>Benign tumours tend to grow slowly and do not spread, while malignant tumours tend to grow much faster and have the potential to invade and destroy adjacent tissues, and spread to the rest of the body.</p>
        <p>This is the structure of the BreakHis 1.0 dataset from the source website:</p>
      </article>
      <BreakHisTable />
      <p class="prose mx-40 mt-10">This is the ratio of samples which are benign to those which are malignant:</p>
      <BarGraph data={originalDataDistribution} />
      <p class="prose mx-40">The dataset is moderately imbalanced and biased towards the over-represented Malignant class as the ratio of Benign : Malignant samples is 31.4 : 68.6. The dataset should be less skewed towards the Malignant samples.</p>
      <article class="prose mx-40 mt-10">
        <p>And this is a more detailed breakdown I got by running a shell script to count the number of samples for each tumour type and magnification in the dataset:</p>
      </article>
      <DetailedBreakHisTable />
      <article class="prose mx-40 mt-10">
        <p>There are four types of benign tumours in the dataset: adenosis (A), fibroadenoma (F), phyllodes tumor (PT), and tubular adenona (TA);  and four malignant tumors (breast cancer): carcinoma (DC), lobular carcinoma (LC), mucinous carcinoma (MC) and papillary carcinoma (PC).</p>
        <h3>Benign Tumours</h3>
        <p>*These are some descriptions I found on the internet which I hope can provide context for the dataset, but do not take it as an accurate source of medical information.*</p>
        <p>Images are from the BreakHis 1.0 dataset.</p>
      </article>
      <div class="grid grid-cols-2 gap-4 mx-40 mt-10">
        <Card title="Adenosis" description="Tumour caused by enlarged milk-producing glands." imageSrc={Adenosis}/>
        <Card title="Fibroadenoma" description="Tumour caused by excessive fibrous tissue developing in the glands." imageSrc={Fibroadenoma}/>
        <Card title="Phyllodes Tumour" description="Tumour that develops in the connective tissue of the breast." imageSrc={Phyllodes}/>
        <Card title="Tubular Adenoma" description="Tumour characterised by tube-like structures visible under a microscope." imageSrc={TubularAdenoma}/>
      </div>
      <article class="prose mx-40 mt-10">
        <h3>Malignant Tumours</h3>
      </article>
      <div class="grid grid-cols-2 gap-4 mx-40 mt-10">
        <Card title="Ductal Carcinoma" description="Cancer originating from the cells lining the milk ducts." imageSrc={DuctalCarcinoma}/>
        <Card title="Lobular Carcinoma" description="Cancer originating from cells in milk-producing glands." imageSrc={LobularCarcinoma}/>
        <Card title="Mucinous Carcinoma" description="Characterised by cancerous cells which are surrounded by a substance called mucin, a protein found in mucus." imageSrc={MucinousCarcinoma}/>
        <Card title="Papillary Carcinoma" description="Characterised by the presence of finger-like projections when cancerous cells are viewed under a microsope." imageSrc={PapillaryCarcinoma}/>
      </div>
      <article class="prose mx-40 mt-10">

      </article>
      <article class="prose mx-40 mt-10">
        <p>After performing EDA, the next step is to prepare the data to train and test the neural networks!</p>
      </article>
    </div>

  )
}
export default EDA;