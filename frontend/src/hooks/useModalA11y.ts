import { useEffect, useRef } from "react";

const FOCUSABLE_SELECTOR =
  'a[href], button:not([disabled]), textarea:not([disabled]), input:not([disabled]), select:not([disabled]), [tabindex]:not([tabindex="-1"])';

/**
 * Shared modal accessibility behavior for this app's fixed-overlay dialogs:
 * focuses the first focusable element on mount, traps Tab/Shift+Tab within
 * the container while open (re-queried live on every Tab press, so it stays
 * correct across modals whose content changes shape after mount — e.g. a
 * revealed override form, or a swapped result view), closes on Escape, and
 * returns focus to whatever triggered the modal once it unmounts.
 *
 * `onEscape` is read fresh on every render (via a ref), so a caller can pass
 * a different callback per render — e.g. a no-op while a request is in
 * flight and no Cancel affordance is visibly available, matching whatever
 * the modal's own Cancel/Close button is currently allowed to do.
 *
 * Attach the returned ref to the modal's outer `role="dialog"` element.
 */
export function useModalA11y(onEscape: () => void) {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const escapeRef = useRef(onEscape);
  escapeRef.current = onEscape;

  useEffect(() => {
    const trigger = document.activeElement as HTMLElement | null;
    const container = containerRef.current;

    // No `offsetParent`/visibility filtering: every modal in this app
    // conditionally *mounts* its focusable elements rather than hiding them
    // via CSS while staying in the DOM, so a plain query is both correct
    // and avoids jsdom's lack of real layout (offsetParent is always null
    // there, which would otherwise break this in every test).
    const getFocusable = () =>
      container ? Array.from(container.querySelectorAll<HTMLElement>(FOCUSABLE_SELECTOR)) : [];

    (getFocusable()[0] ?? container)?.focus();

    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === "Escape") {
        e.stopPropagation();
        escapeRef.current();
        return;
      }
      if (e.key !== "Tab") return;
      const items = getFocusable();
      if (items.length === 0) return;
      const first = items[0];
      const last = items[items.length - 1];
      if (e.shiftKey && document.activeElement === first) {
        e.preventDefault();
        last.focus();
      } else if (!e.shiftKey && document.activeElement === last) {
        e.preventDefault();
        first.focus();
      }
    };

    document.addEventListener("keydown", handleKeyDown, true);
    return () => {
      document.removeEventListener("keydown", handleKeyDown, true);
      if (trigger && document.contains(trigger) && typeof trigger.focus === "function") {
        trigger.focus();
      }
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  return containerRef;
}
