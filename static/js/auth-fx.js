/* Sign-in / register pages: the live sound field behind the brand panel and a short entrance. */
import { mountSoundField } from "./soundfield.js";

const full = !window.SSMotion || window.SSMotion.full;
const gsap = window.gsap;
const field = document.getElementById("auth-field");
const readout = document.querySelector("[data-auth-event]");
if (field) {
  mountSoundField(field, {
    animate: full, interactive: full, rows: 80, cols: 120, background: "#0c0d0b", mid: "#ece6d9", hot: "#ff5a1f",
    onEvent: (name, on) => { if (readout) readout.textContent = on ? name.toLowerCase() : "background noise"; },
  });
}
if (window.SSMotion) window.SSMotion.prompt();
if (full && gsap) {
  const title = document.querySelector("[data-auth-title]");
  if (title && window.SplitText) {
    const split = window.SplitText.create(title, { type: "lines,chars", mask: "lines" });
    gsap.from(split.chars, { yPercent: 110, duration: 0.9, ease: "power4.out", stagger: 0.015, delay: 0.1 });
  }
  gsap.from("[data-auth-in]", { opacity: 0, y: 16, duration: 0.8, ease: "power3.out", stagger: 0.08, delay: 0.3 });
  const card = document.querySelector("[data-auth-card]");
  if (card) gsap.from(card.children, { opacity: 0, y: 22, duration: 0.8, ease: "power3.out", stagger: 0.07, delay: 0.15 });
}
