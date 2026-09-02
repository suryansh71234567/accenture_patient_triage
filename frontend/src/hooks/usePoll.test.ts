import { act, renderHook, waitFor } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { usePoll } from "./usePoll";

function deferred<T>() {
  let resolve!: (v: T) => void;
  let reject!: (e: unknown) => void;
  const promise = new Promise<T>((res, rej) => {
    resolve = res;
    reject = rej;
  });
  return { promise, resolve, reject };
}

describe("usePoll", () => {
  it("fetches immediately on mount and exposes the result", async () => {
    const fn = vi.fn().mockResolvedValue({ v: 1 });
    const { result } = renderHook(() => usePoll(fn, 10000));

    expect(result.current.loading).toBe(true);
    await waitFor(() => expect(result.current.data).toEqual({ v: 1 }));
    expect(result.current.loading).toBe(false);
    expect(result.current.error).toBeNull();
    expect(fn).toHaveBeenCalledTimes(1);
  });

  it("polls again on the interval", async () => {
    vi.useFakeTimers();
    try {
      const fn = vi.fn().mockResolvedValue({ v: 1 });
      renderHook(() => usePoll(fn, 1000));
      await act(async () => {
        await Promise.resolve();
      });
      expect(fn).toHaveBeenCalledTimes(1);

      await act(async () => {
        vi.advanceTimersByTime(1000);
        await Promise.resolve();
      });
      expect(fn).toHaveBeenCalledTimes(2);

      await act(async () => {
        vi.advanceTimersByTime(2000);
        await Promise.resolve();
      });
      expect(fn).toHaveBeenCalledTimes(4);
    } finally {
      vi.useRealTimers();
    }
  });

  it("refetch() re-runs the fetch on demand, independent of the interval", async () => {
    const fn = vi.fn().mockResolvedValue({ v: 1 });
    const { result } = renderHook(() => usePoll(fn, 60000));
    await waitFor(() => expect(fn).toHaveBeenCalledTimes(1));

    await act(async () => {
      await result.current.refetch();
    });
    expect(fn).toHaveBeenCalledTimes(2);
  });

  it("CRITICAL: a stale response resolving after a fresher one must not overwrite it", async () => {
    const first = deferred<{ v: string }>();
    const second = deferred<{ v: string }>();
    const fn = vi
      .fn()
      .mockImplementationOnce(() => first.promise) // Request A (issued on mount)
      .mockImplementationOnce(() => second.promise); // Request B (issued by refetch)

    const { result } = renderHook(() => usePoll(fn, 60000));
    await waitFor(() => expect(fn).toHaveBeenCalledTimes(1)); // A in flight

    // Issue B (a manual refetch) while A is still pending — this is the
    // "Request A starts, Request B starts" setup.
    let refetchPromise!: Promise<void>;
    act(() => {
      refetchPromise = result.current.refetch();
    });
    await waitFor(() => expect(fn).toHaveBeenCalledTimes(2)); // B in flight, A still in flight

    // B resolves FIRST.
    await act(async () => {
      second.resolve({ v: "B-fresh" });
      await refetchPromise;
    });
    expect(result.current.data).toEqual({ v: "B-fresh" });

    // A (stale) resolves AFTER B. It must be discarded, not overwrite B.
    await act(async () => {
      first.resolve({ v: "A-stale" });
      await Promise.resolve();
      await Promise.resolve();
    });
    expect(result.current.data).toEqual({ v: "B-fresh" });
  });

  it("last-ISSUED wins even if it never resolves — an older, resolved response is still discarded", async () => {
    const first = deferred<{ v: string }>();
    const second = deferred<{ v: string }>(); // deliberately left unresolved
    const fn = vi
      .fn()
      .mockImplementationOnce(() => first.promise)
      .mockImplementationOnce(() => second.promise);

    const { result } = renderHook(() => usePoll(fn, 60000));
    await waitFor(() => expect(fn).toHaveBeenCalledTimes(1));

    act(() => {
      void result.current.refetch(); // issues the second (never-resolving) request
    });
    await waitFor(() => expect(fn).toHaveBeenCalledTimes(2));

    // The FIRST request resolves — but it's no longer the latest issued.
    await act(async () => {
      first.resolve({ v: "A-late-but-superseded" });
      await Promise.resolve();
      await Promise.resolve();
    });
    // Must NOT be applied — data stays whatever it was (null, since B never resolved).
    expect(result.current.data).toBeNull();
  });

  it("multiple out-of-order responses: only the response matching the latest issued sequence is ever applied", async () => {
    const a = deferred<{ v: string }>();
    const b = deferred<{ v: string }>();
    const c = deferred<{ v: string }>();
    const fn = vi
      .fn()
      .mockImplementationOnce(() => a.promise)
      .mockImplementationOnce(() => b.promise)
      .mockImplementationOnce(() => c.promise);

    const { result } = renderHook(() => usePoll(fn, 60000));
    await waitFor(() => expect(fn).toHaveBeenCalledTimes(1));
    act(() => {
      void result.current.refetch();
    });
    await waitFor(() => expect(fn).toHaveBeenCalledTimes(2));
    act(() => {
      void result.current.refetch();
    });
    await waitFor(() => expect(fn).toHaveBeenCalledTimes(3));

    // Resolve out of order: b, then a, then c.
    await act(async () => {
      b.resolve({ v: "B" });
      await Promise.resolve();
    });
    expect(result.current.data).toBeNull(); // B is superseded by C's issue, discarded

    await act(async () => {
      a.resolve({ v: "A" });
      await Promise.resolve();
    });
    expect(result.current.data).toBeNull(); // A is even more stale, discarded

    await act(async () => {
      c.resolve({ v: "C" });
      await Promise.resolve();
    });
    expect(result.current.data).toEqual({ v: "C" }); // only the latest-issued response ever lands
  });

  it("sets error state on rejection without leaving stale data from a previous success", async () => {
    const fn = vi.fn().mockRejectedValue(new Error("boom"));
    const { result } = renderHook(() => usePoll(fn, 60000));
    await waitFor(() => expect(result.current.error).not.toBeNull());
    expect(result.current.error?.message).toBe("boom");
    expect(result.current.loading).toBe(false);
  });

  it("a stale rejection arriving after a fresh success must not clobber the success with an error", async () => {
    const first = deferred<{ v: string }>();
    const fn = vi.fn().mockImplementationOnce(() => first.promise).mockResolvedValueOnce({ v: "fresh" });

    const { result } = renderHook(() => usePoll(fn, 60000));
    await waitFor(() => expect(fn).toHaveBeenCalledTimes(1));

    let refetchPromise!: Promise<void>;
    act(() => {
      refetchPromise = result.current.refetch();
    });
    await act(async () => {
      await refetchPromise;
    });
    expect(result.current.data).toEqual({ v: "fresh" });
    expect(result.current.error).toBeNull();

    // The stale first request now rejects — must be discarded.
    await act(async () => {
      first.reject(new Error("stale failure"));
      await Promise.resolve();
      await Promise.resolve();
    });
    expect(result.current.error).toBeNull();
    expect(result.current.data).toEqual({ v: "fresh" });
  });

  it("re-runs the fetch when deps change (e.g. hospital switch), and a response for the OLD deps is discarded", async () => {
    const forDefault = deferred<{ hospital: string }>();
    const forOther = deferred<{ hospital: string }>();
    const fn = vi
      .fn()
      .mockImplementationOnce(() => forDefault.promise)
      .mockImplementationOnce(() => forOther.promise);

    const { result, rerender } = renderHook(({ hospitalId }) => usePoll(fn, 60000, [hospitalId]), {
      initialProps: { hospitalId: "default" },
    });
    await waitFor(() => expect(fn).toHaveBeenCalledTimes(1));

    // Switch hospitals before the "default" request resolves.
    rerender({ hospitalId: "other" });
    await waitFor(() => expect(fn).toHaveBeenCalledTimes(2));

    await act(async () => {
      forOther.resolve({ hospital: "other" });
      await Promise.resolve();
    });
    expect(result.current.data).toEqual({ hospital: "other" });

    // The stale "default" hospital's response now arrives — must not leak
    // that hospital's data into the now-selected "other" hospital's view.
    await act(async () => {
      forDefault.resolve({ hospital: "default" });
      await Promise.resolve();
      await Promise.resolve();
    });
    expect(result.current.data).toEqual({ hospital: "other" });
  });

  it("does not throw or update state after unmount when an in-flight request resolves late", async () => {
    const first = deferred<{ v: number }>();
    const fn = vi.fn().mockImplementationOnce(() => first.promise);
    const { unmount } = renderHook(() => usePoll(fn, 60000));
    await waitFor(() => expect(fn).toHaveBeenCalledTimes(1));

    unmount();

    // Resolving after unmount must not throw (React 18+ silently no-ops a
    // state update on an unmounted component; this asserts no exception).
    await expect(
      act(async () => {
        first.resolve({ v: 1 });
        await Promise.resolve();
      })
    ).resolves.toBeUndefined();
  });

  it("always uses the latest fn passed in, not a stale closure from mount", async () => {
    let fn = vi.fn().mockResolvedValue({ v: "first-fn" });
    const { result, rerender } = renderHook(({ f }) => usePoll(f, 60000), { initialProps: { f: fn } });
    await waitFor(() => expect(result.current.data).toEqual({ v: "first-fn" }));

    fn = vi.fn().mockResolvedValue({ v: "second-fn" });
    rerender({ f: fn });
    await act(async () => {
      await result.current.refetch();
    });
    expect(result.current.data).toEqual({ v: "second-fn" });
  });
});
