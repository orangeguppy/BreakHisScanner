import React from "react";

function DetailedBreakHisTable() {
  return(
    <div class="overflow-x-auto">
      <table class="table normal w-6/12 mx-20 mt-10">
        <thead>
          <tr>
            <th></th>
            <th colspan="4" class="text-center">Benign</th>
            <th colspan="4" class="text-center">Malignant</th>
          </tr>
          <tr>
            <th>Magnification</th>
            <th>Adenosis</th>
            <th>Fibroadenoma</th>
            <th>Phyllodes T.</th>
            <th>Tubular Ade.</th>
            <th>Ductal Carci.</th>
            <th>Lobular Carci.</th>
            <th>Mucinous Carci.</th>
            <th>Papillary Carci.</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td>40X</td>
            <td>114</td>
            <td>253</td>
            <td>109</td>
            <td>149</td>
            <td>864</td>
            <td>156</td>
            <td>205</td>
            <td>145</td>
          </tr>
          <tr>
            <td>100X</td>
            <td>113</td>
            <td>260</td>
            <td>121</td>
            <td>150</td>
            <td>902</td>
            <td>170</td>
            <td>222</td>
            <td>142</td>
          </tr>
          <tr>
            <td>200X</td>
            <td>111</td>
            <td>264</td>
            <td>108</td>
            <td>140</td>
            <td>896</td>
            <td>163</td>
            <td>196</td>
            <td>135</td>
          </tr>
          <tr>
            <td>400X</td>
            <td>106</td>
            <td>237</td>
            <td>115</td>
            <td>130</td>
            <td>788</td>
            <td>137</td>
            <td>169</td>
            <td>138</td>
          </tr>
          <tr>
            <td>Total Images</td>
            <td>444</td>
            <td>1,014</td>
            <td>453</td>
            <td>569</td>
            <td>3,450</td>
            <td>626</td>
            <td>792</td>
            <td>560</td>
          </tr>
        </tbody>
      </table>
    </div>
  )
}export default DetailedBreakHisTable;