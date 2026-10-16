/* GTM training helper: plays a class's concatenated training samples. */
(function () {
  "use strict";
  const audio = new Audio();
  const label = document.getElementById("trainer-label");
  const time = document.getElementById("trainer-time");
  const vol = document.getElementById("trainer-vol");
  const state = document.querySelector("#trainer .mic-state");
  vol.addEventListener("input", () => { audio.volume = Number(vol.value); });
  document.querySelectorAll("[data-play-class]").forEach((b) => {
    b.addEventListener("click", () => {
      audio.src = b.dataset.playClass;
      audio.currentTime = 0;
      audio.play().then(() => {
        label.textContent = `Playing: ${b.dataset.name}`;
        state.dataset.state = "active";
      }).catch((e) => { label.textContent = `Cannot play: ${e.message}`; });
    });
  });
  document.getElementById("trainer-stop").addEventListener("click", () => { audio.pause(); state.dataset.state = "available"; label.textContent = "Stopped"; });
  audio.addEventListener("timeupdate", () => { time.textContent = `${window.SS.fmt.tc(audio.currentTime)} / ${window.SS.fmt.tc(audio.duration || 0)}`; });
  audio.addEventListener("ended", () => { state.dataset.state = "available"; label.textContent = "Finished - extract the samples in Teachable Machine"; });
})();
