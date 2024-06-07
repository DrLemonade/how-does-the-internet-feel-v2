const express = require("express");
const cors = require("cors");
const app = express();
const PORT = process.env.PORT || 5000;

app.use(express.json());
app.use(cors());

app.post("/", (req, res) => {
    const { search_term } = req.body;
    const result = Math.floor(Math.random() * 6);

    return res.status(200).send({ category: result });
});

app.listen(PORT, () => {
    console.log("Server is running on port " + PORT);
});
