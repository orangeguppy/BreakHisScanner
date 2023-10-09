import './App.css';
import Navbar from "./components/Navbar.js";
import Plan from "./pages/Plan.js";
import EDA from "./pages/EDA.js";
import DataPrep from "./pages/DataPrep.js";
import { Routes, Route } from "react-router-dom";
import { MathJaxContext } from "better-react-mathjax";

function App() {
  return (
  <>
    <Navbar/>
    <MathJaxContext>
        <Routes>
          <Route path="/" element={<Plan/>} />
          <Route path="/eda" element={<EDA/>} />
          <Route path="/data-prep" element={<DataPrep/>} />
        </Routes>
    </MathJaxContext>
  </>
  );
}
export default App;
