import "@testing-library/jest-dom/vitest";

// jsdom doesn't implement Element.scrollTo — ChatDock (unrelated to this
// project's changes) calls it on every history update.
if (!Element.prototype.scrollTo) {
  Element.prototype.scrollTo = () => {};
}
