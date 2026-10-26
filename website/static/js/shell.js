import { $, formatWord } from "./ui.js";

document.querySelectorAll("[data-format-word]").forEach((el) => {
  el.textContent = formatWord(el.dataset.formatWord);
});
const menu = $("menu-button");
const sidebar = $("sidebar");
function closeMenu() {
  sidebar.classList.remove("is-open");
  $("nav-scrim").hidden = true;
  menu.setAttribute("aria-expanded", "false");
  sidebar.inert = matchMedia("(max-width: 780px)").matches;
}
function syncMenu() {
  closeMenu();
  sidebar.inert = matchMedia("(max-width: 780px)").matches;
}
menu.addEventListener("click", () => {
  const open = menu.getAttribute("aria-expanded") !== "true";
  sidebar.classList.toggle("is-open", open);
  sidebar.inert = !open;
  $("nav-scrim").hidden = !open;
  menu.setAttribute("aria-expanded", String(open));
});
$("nav-scrim").addEventListener("click", closeMenu);
matchMedia("(max-width: 780px)").addEventListener("change", syncMenu);
syncMenu();
const dialog = $("help-dialog");
document.querySelectorAll("[data-open-help]").forEach((button) => button.addEventListener("click", () => dialog.showModal()));
document.querySelectorAll("[data-close-help]").forEach((button) => button.addEventListener("click", () => dialog.close()));
dialog.addEventListener("click", (event) => { if (event.target === dialog) {
  const box = dialog.getBoundingClientRect();
  if (event.clientX < box.left || event.clientX > box.right || event.clientY < box.top || event.clientY > box.bottom) dialog.close();
}});
document.addEventListener("keydown", (event) => { if (event.key === "Escape") closeMenu(); });
