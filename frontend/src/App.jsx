import React from "react";
import Dashboard from "./components/Dashboard";
import FluidSmoke from "./components/FluidSmoke";
import "./style.css";

function App() {
  return (
    <div className="app-container">
      {/* This component renders the canvas. 
          It stays fixed in the background via CSS. 
      */}
      <FluidSmoke />

      {/* Your main content layer. 
          The style.css ensures this has a higher z-index. 
      */}
      <Dashboard />
    </div>
  );
}

export default App;