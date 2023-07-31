import React from "react";
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend } from 'recharts';


function BarGraph(props) {
  return (
    <div class="p-8 mx-40">
      <BarChart width={300} height={250} data={props.data}>
        <CartesianGrid strokeDasharray="3 3" />
        <XAxis dataKey="name" />
        <YAxis />
        <Tooltip />
        <Legend />
        <Bar dataKey="Benign" fill="#DBE1FF" />
        <Bar dataKey="Malignant" fill="#DBD4ED" />
      </BarChart>
    </div>
  );
} export default BarGraph;