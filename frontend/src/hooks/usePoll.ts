import { useCallback, useEffect, useRef, useState } from "react";

/**
 * Fetches `fn()` immediately, then again every `intervalMs`, and whenever
 * `deps` change or `refetch()` is called manually. Used for hospital state /
 * live simulation dashboard views so they feel "live" without a websocket.
 */
export function usePoll<T>(
  fn: () => Promise<T>,
  intervalMs: number,
  deps: unknown[] = []
) {
  const [data, setData] = useState<T | null>(null);
  const [error, setError] = useState<Error | null>(null);
  const [loading, setLoading] = useState(true);
  const fnRef = useRef(fn);
  fnRef.current = fn;
  // Discards a response that resolves after a newer fetch has already been
  // issued — by the interval or by a manual refetch() — so a slow, stale
  // response can never overwrite fresher state that already landed.
  const seqRef = useRef(0);

  const refetch = useCallback(async () => {
    const seq = ++seqRef.current;
    try {
      const result = await fnRef.current();
      if (seq === seqRef.current) {
        setData(result);
        setError(null);
      }
    } catch (err) {
      if (seq === seqRef.current) setError(err as Error);
    } finally {
      if (seq === seqRef.current) setLoading(false);
    }
  }, []);

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    const run = async () => {
      const seq = ++seqRef.current;
      try {
        const result = await fnRef.current();
        if (!cancelled && seq === seqRef.current) {
          setData(result);
          setError(null);
        }
      } catch (err) {
        if (!cancelled && seq === seqRef.current) setError(err as Error);
      } finally {
        if (!cancelled && seq === seqRef.current) setLoading(false);
      }
    };
    run();
    const id = setInterval(run, intervalMs);
    return () => {
      cancelled = true;
      clearInterval(id);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, deps);

  return { data, error, loading, refetch };
}
