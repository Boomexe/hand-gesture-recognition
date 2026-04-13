// chrome.action.onClicked.addListener(async (tab) => {
//   await chrome.scripting.executeScript({
//     target: { tabId: tab.id },
//     world: "MAIN",
//     function: () => {
//       console.log("Hello from the Google Extension!");
//       console.log("" + document.querySelector(".video-stream").currentTime);
//     },
//   });
// });

function getCurrentTime() {
  //   console.log("" + document.querySelector(".video-stream").currentTime);
  return document.querySelector(".video-stream").currentTime;
}

getCurrentTime();

word_list = [];

word = "";
first = "";
second = "";

document.addEventListener("keydown", (event) => {
  if (event.key === "Escape") {
    console.log("word list: " + word_list);
    navigator.clipboard.writeText(word_list);
    word_list = [];
  }

  if (event.key === "s") {
    if (first === "") {
      first = getCurrentTime();
    } else if (second === "") {
      second = getCurrentTime();

      console.log("first: " + first);

      word = prompt("Enter the word for the timestamp: ", word);

      timestamp = `{"word": "${word}", "timestamp_start": "${secondsToTimestamp(first)}","timestamp_end": "${secondsToTimestamp(second)}}"`;

      word_list.push(timestamp);

      console.log(word_list);
      first = "";
      second = "";
    }
  }
});

function secondsToTimestamp(seconds) {
  const totalMs = Math.round(seconds * 1000);

  const ms = totalMs % 1000;
  const totalSeconds = Math.floor(totalMs / 1000);

  const s = totalSeconds % 60;
  const totalMinutes = Math.floor(totalSeconds / 60);

  const m = totalMinutes % 60;
  const h = Math.floor(totalMinutes / 60);

  const pad = (n, len = 2) => String(n).padStart(len, "0");

  return `${pad(h)}:${pad(m)}:${pad(s)}.${pad(ms, 3)}`;
}
