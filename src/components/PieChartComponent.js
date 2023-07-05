import React, { PureComponent } from "react";
import { PieChart, Pie, Legend } from "recharts";

function PieChartComponent(props) {

  return(
    <PieChart width={2900} height={400}>
      <Legend
        iconType="circle"
        layout="vertical"
        verticalAlign="middle"
      />
      <Pie
        data={props.data}
        cx={120}
        cy={200}
        innerRadius={20}
        outerRadius={80}
        fill="#8884d8"
        dataKey="value"
      />
    </PieChart>
  );
} export default PieChartComponent;