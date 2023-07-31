import React from "react";

function SampleResult() {
  return(
      <table class="table">
        <thead>
          <tr>
            <th>Model</th>
            <th>VGG19</th>
            <th>InceptionV4</th>
            <th>ResNet152</th>
            <th>DenseNet201</th>
            <th>Actual</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td>Result</td>
            <td>Malignant</td>
            <td>Malignant</td>
            <td>Benign</td>
            <td>Benign</td>
            <td>Benign</td>
          </tr>
        </tbody>
      </table>
  )
}export default SampleResult;