import { $, formatWord } from "./ui.js";

const categories = {
  everyday: new Set(["HELLO", "THANKYOU", "PLEASE", "YES", "NO", "MORE", "HELP", "GOOD", "BAD", "FINISH", "WANT", "NEED", "WATER", "DRINK", "EAT", "HOME"]),
  people: new Set(["FAMILY", "FATHER", "MOTHER", "FRIEND", "ME", "MY", "YOU", "YOUR", "WE", "WHO"]),
  questions: new Set(["HOW", "WHAT", "WHEN", "WHERE", "WHO", "WHY"]),
};
const tiles = [...document.querySelectorAll(".gesture-tile")];
let category = "all";
function filter() {
  const query = $("gesture-search").value.trim().toLowerCase().replace(/[^a-z0-9]/g, "");
  let count = 0;
  for (const tile of tiles) {
    const word = tile.dataset.word;
    const text = (word + formatWord(word)).toLowerCase().replace(/[^a-z0-9]/g, "");
    const matchesCategory = category === "all" || categories[category].has(word.replace(/\d+$/, ""));
    tile.hidden = !matchesCategory || !text.includes(query);
    if (!tile.hidden) count++;
  }
  $("library-count").textContent = count + (count === 1 ? " gesture" : " gestures");
  $("library-empty").hidden = count > 0;
  $("gesture-grid").hidden = count === 0;
}
$("gesture-search").addEventListener("input", filter);
document.querySelectorAll("[data-category]").forEach((button) => button.addEventListener("click", () => {
  category = button.dataset.category;
  document.querySelectorAll("[data-category]").forEach((item) => {
    item.classList.toggle("active", item === button);
    item.setAttribute("aria-pressed", String(item === button));
  });
  filter();
}));
$("reset-search").addEventListener("click", () => {
  $("gesture-search").value = "";
  document.querySelector('[data-category="all"]').click();
  $("gesture-search").focus();
});
document.addEventListener("keydown", (event) => {
  if (event.key === "/" && !["INPUT", "TEXTAREA"].includes(document.activeElement.tagName) && !$("help-dialog").open) {
    event.preventDefault();
    $("gesture-search").focus();
  }
});
