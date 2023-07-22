import './App.css';
import Navbar from "./components/Navbar.js";
import Home from "./pages/Home.js";
import Plan from "./pages/Plan.js";
import EDA from "./pages/EDA.js";
import DataPrep from "./pages/DataPrep.js";
import { Routes, Route } from "react-router-dom";
import { MathJaxContext } from "better-react-mathjax";

function App() {
  return (
    <MathJaxContext>
      <Navbar />
        <Routes>
          <Route path="/" element={<Home/>} />
          <Route path="/plan" element={<Plan/>} />
          <Route path="/eda" element={<EDA/>} />
          <Route path="/data-prep" element={<DataPrep/>} />
        </Routes>
    </MathJaxContext>
  );
}

export default App;
