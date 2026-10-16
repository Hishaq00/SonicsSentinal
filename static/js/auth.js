/* Sign-in page: demo-account buttons fill the form. */
(function () {
  "use strict";
  document.querySelectorAll("[data-demo]").forEach((b) => {
    b.addEventListener("click", () => {
      const u = document.getElementById("username");
      const p = document.getElementById("password");
      if (u && p) { u.value = b.dataset.demo; p.value = b.dataset.pw; p.focus(); }
    });
  });
})();
