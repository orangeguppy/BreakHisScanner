import React, { useState } from "react";
import SampleResult from "./SampleResult.js";

import Adenosis from '../assets/images/tumour_samples/benign/adenosis_100x.png';
import Fibroadenoma from '../assets/images/tumour_samples/benign/fibroadenoma_100x.png';
import Phyllodes from '../assets/images/tumour_samples/benign/phyllodes_100x.png';
import TubularAdenoma from '../assets/images/tumour_samples/benign/tubular_adenoma_100x.png';

import DuctalCarcinoma from "../assets/images/tumour_samples/malignant/ductal_carcinoma_100x.png";
import LobularCarcinoma from "../assets/images/tumour_samples/malignant/lobular_carcinoma_100x.png";
import MucinousCarcinoma from "../assets/images/tumour_samples/malignant/mucinous_carcinoma_100x.png";
import PapillaryCarcinoma from "../assets/images/tumour_samples/malignant/papillary_carcinoma_100x.png";

function SampleDisplayer(props) {
  const [selectedSample, setSelectedSample] = useState(Fibroadenoma);

  const images = [
    Adenosis, Fibroadenoma, Phyllodes, TubularAdenoma, DuctalCarcinoma, LobularCarcinoma, MucinousCarcinoma, PapillaryCarcinoma
  ]

  function selectRandomSample() {
    const randomIndex = Math.floor(Math.random() * images.length);
    setSelectedSample(images[randomIndex]);
    sendRequest(selectedSample);
  }

  async function sendRequest(selectedImage){
    const formData = new FormData();
    formData.append('image', selectedImage);

    const response = await fetch("http://127.0.0.1:9696/densenet201", {
      method: "POST",
      headers: {'Content-Type': 'application/json'},
      body: formData
    })
    .then(res => res.json())
    .then(data => {
      console.log(data)
    })
  }

  return(
    <div class="card card-compact w-50 bg-base-100 shadow-xl pb-20 px-20">
      <h1>{props.imgPath}</h1>
      <figure><img src={selectedSample} alt="Dataset Sample" height="500" width="500"/></figure>
      <div class="card-body">
        <button class="btn btn-neutral" onClick={selectRandomSample} >Random!</button>
        <h2 class="card-title">{props.title}</h2>
        <p>{props.description}</p>
        <SampleResult />
        <div class="card-actions justify-end">
        </div>
      </div>
    </div>
  )
}export default SampleDisplayer;