import React from "react";
import { ImSpinner8 } from "react-icons/im";

import "./LoadingSpinner.css";

const LoadingSpinner = ({ size }) => {
    return (
        <div className="spinnerContainer">
            <ImSpinner8 className="spinner" color="white" size={size} />
        </div>
    );
};

export default LoadingSpinner;
