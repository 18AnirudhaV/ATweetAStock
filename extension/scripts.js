// Handles your frontend UI logic.
const pos = "👍";
const neg = "👎";
const neu = "🤷";

var input = window.prompt("");
if (input == "pos") {
    document.body.innerHTML = document.body.innerHTML.replace('emoji', pos);
} else if (input == "neg") {
    document.body.innerHTML = document.body.innerHTML.replace('emoji', neg);
} else {
    document.body.innerHTML = document.body.innerHTML.replace('emoji', neu);
}


