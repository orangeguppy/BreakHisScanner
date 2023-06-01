import React from "react";

function BreakHisTable() {
  return(
    <div class="overflow-x-auto">
      <table class="table normal w-6/12 mx-40 mt-10">
        <thead>
          <tr>
            <th>Magnification</th>
            <th>Benign</th>
            <th>Malignant</th>
            <th>Total</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td>40X</td>
            <td>652</td>
            <td>1,370</td>
            <td>1,995</td>
          </tr>
          <tr>
            <td>100X</td>
            <td>644</td>
            <td>1,437</td>
            <td>2,081</td>
          </tr>
          <tr>
            <td>200X</td>
            <td>623</td>
            <td>1,390</td>
            <td>2,013</td>
          </tr>
          <tr>
            <td>400X</td>
            <td>588</td>
            <td>1,232</td>
            <td>1,820</td>
          </tr>
          <tr>
            <td>Total Images</td>
            <td>2,480</td>
            <td>5,429</td>
            <td>7,909</td>
          </tr>
        </tbody>
      </table>
    </div>
  )
}export default BreakHisTable;