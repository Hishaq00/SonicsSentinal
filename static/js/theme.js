/* Display preferences applied before the page paints (avoids a flash):
 * colour theme (light / dark) and motion (full / reduced).
 * Motion defaults to full. When the operating system asks for reduced motion
 * and the visitor has not chosen yet, SSMotion.prompt() offers the choice. */
(function () {
  var root = document.documentElement;
  var store = { get: function (k) { try { return localStorage.getItem(k); } catch (e) { return null; } },
                set: function (k, v) { try { localStorage.setItem(k, v); } catch (e) { /* storage unavailable */ } } };
  var t = store.get("ss-theme");
  if (t === "light" || t === "dark") root.setAttribute("data-theme", t);

  var osReduce = !!(window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches);
  var chosen = store.get("ss-motion");
  var mode = chosen === "reduced" || chosen === "full" ? chosen : "full";
  root.setAttribute("data-motion", mode);
  if (mode === "full") root.classList.add("m-pending"); // motion.js reveals the page content

  window.SSMotion = {
    full: mode === "full",
    osReduce: osReduce,
    chosen: chosen === "reduced" || chosen === "full",
    set: function (value, reload) {
      store.set("ss-motion", value === "reduced" ? "reduced" : "full");
      if (reload !== false) window.location.reload();
    },
    toggle: function () { this.set(this.full ? "reduced" : "full"); },
    /* One-time question when the system prefers reduced motion. */
    prompt: function () {
      if (!osReduce || this.chosen || document.querySelector(".ss-motion-ask")) return;
      var box = document.createElement("div");
      box.className = "ss-motion-ask";
      box.setAttribute("role", "dialog");
      box.setAttribute("aria-label", "Animation preference");
      box.innerHTML = '<p><b>Your device asks websites to reduce motion.</b> This site uses animation to explain how it works.</p>' +
        '<div><button type="button" data-keep>Keep animations</button><button type="button" data-reduce>Reduce motion</button></div>';
      box.querySelector("[data-keep]").addEventListener("click", function () { window.SSMotion.set("full", false); window.SSMotion.chosen = true; box.remove(); });
      box.querySelector("[data-reduce]").addEventListener("click", function () { window.SSMotion.set("reduced"); });
      document.body.appendChild(box);
    }
  };
})();
