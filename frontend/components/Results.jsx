import React from "react";

import "./Results.css";
import LoadingSpinner from "./LoadingSpinner";

const Results = ({ loading, result }) => {
    const categories = ["Surprise", "Fear", "Anger", "Love", "Joy", "Sadness"];

    return (
        <div className="codeBackground">
            {!loading ? (
                <>
                    {"{"}
                    <p className="result">
                        <span className="indentation">
                            <span className="attribute">Emotion</span>:{" "}
                            <span className="value">{categories[result]}</span>,
                        </span>
                        {/* <br />
                        <span className="indentation">
                            <span className="attribute">Positive</span>:{" "}
                            <span className="value">
                                {results.num_positive}
                            </span>
                            ,
                        </span>
                        <br />
                        <span className="indentation">
                            <span className="attribute">Negative</span>:{" "}
                            <span className="value">
                                {results.num_negative}
                            </span>
                            ,
                        </span>
                        <br />
                        <span className="indentation">
                            <span className="attribute">% Positive</span>:{" "}
                            <span className="value">
                                {results.percent_positive}
                            </span>
                            ,
                        </span>
                        <br />
                        <span className="indentation">
                            <span className="attribute">% Negative</span>:{" "}
                            <span className="value">
                                {results.percent_negative}
                            </span>
                            ,
                        </span> */}
                    </p>
                    {"}"}
                </>
            ) : (
                <LoadingSpinner size={25} />
            )}
        </div>
    );
};

export default Results;
