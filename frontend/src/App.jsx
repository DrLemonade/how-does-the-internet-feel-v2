import { useState } from "react";
import "./App.css";
import Results from "../components/Results";
import axios from "axios";

function App() {
    const [result, setResult] = useState(-1);
    const [term, setTerm] = useState("");
    // const [status, setStatus] = useState("");
    const [loading, setLoading] = useState(false);

    const submit = async () => {
        if (term.trim() !== "") {
            setLoading(true);
            setResult(-1);
            const res = await axios.post(
                "http://localhost:5000/",
                { search_term: term },
                { headers: { "Content-Type": "application/json" } }
            );

            console.log(res.data);
            setResult(res.data.category);
            setLoading(false);
        }
        setTerm("");
    };

    // const getStatus = (taskID) => {
    //     fetch(`http://localhost:5000/${taskID}`, {
    //         method: "GET",
    //         headers: {
    //             "Content-Type": "application/json",
    //         },
    //     })
    //         .then((response) => response.json())
    //         .then((res) => {
    //             if (res.task_status == "SUCCESS") {
    //                 setLoading(false);
    //                 setStatus("");
    //                 let results = res.task_result;
    //                 console.log(results);
    //                 return setResult(results);
    //             } else if (res.task_status == "FAILURE") {
    //                 setStatus("Error");
    //                 return;
    //             } else {
    //                 setStatus("Loading...");
    //                 setTimeout(function () {
    //                     getStatus(res.task_id);
    //                 }, 1000);
    //             }
    //         })
    //         .catch((err) => console.log(err));
    // };

    return (
        <div className="container">
            <header className="header">Sentiment Analysis</header>
            <div className="idk">
                {/* Enter a search term to analyze the emotion present in the text. */}
            </div>
            <div className="main">
                <input
                    className="input"
                    type="text"
                    value={term}
                    onChange={(e) => setTerm(e.target.value)}
                    onKeyDown={(e) => {
                        if (e.key === "Enter") submit();
                    }}
                    placeholder="Enter a search term"
                />
                <button className="submit" onClick={submit}>
                    Submit
                </button>
            </div>
            <div className="results">
                <Results loading={loading} result={result} />
            </div>

            <footer className="footer">Made by Dante</footer>
        </div>
    );
}

export default App;
