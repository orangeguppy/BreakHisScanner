import logo from './logo.svg';
import './App.css';
import Navbar from "./components/Navbar.js";
import Home from "./pages/Home.js";
import EDA from "./pages/EDA.js";
import { BrowserRouter as Router, Routes, Route, Link } from "react-router-dom";

function App() {
  return (
    <>
      <Navbar />
        <Routes>
          <Route path="/" element={<Home/>} />
          <Route path="/eda" element={<EDA/>} />
        </Routes>
    </>
  );
}

export default App;
