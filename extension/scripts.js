// Handles your frontend UI logic.

var input = prompt();
if (input == "pos") {
    document.getElementById("id01").innerHTML = "Social Sentiment: 👍";
} else if (input == "neg") {
    document.getElementById("id01").innerHTML = "Social Sentiment: 👎";
} else {
    document.getElementById("id01").innerHTML = "Social Sentiment: 🤷";
}
