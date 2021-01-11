import React, { Component } from "react";
import WellCompletions from "../lib/components/WellCompletions";

const data = require("./example-data/well-completions.json");

class WellCompletionsDemo extends Component {
    render() {
        return <WellCompletions id="someid" data={data} />;
    }
}

export default WellCompletionsDemo;
