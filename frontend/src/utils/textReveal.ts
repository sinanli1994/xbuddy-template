/** Presentation only: never delays SSE consumption or changes canonical messages. */
export interface RevealClock {
  now: () => number;
  request: (callback: () => void) => number;
  cancel: (id: number) => void;
}

export function startTextReveal(text: string, paint: (visible: string) => void, clock: RevealClock) {
  const segments = Array.from(new Intl.Segmenter(undefined, { granularity: 'grapheme' }).segment(text),
    part => part.segment);
  // Match the brisk token stream without making validated replies appear at once.
  const duration = Math.min(2000, Math.max(150, segments.length * 4));
  const started = clock.now();
  let frame: number | null = null;
  let cancelled = false;
  let lastCount = -1;
  const tick = () => {
    if (cancelled) return;
    const count = Math.min(segments.length, Math.floor(segments.length * (clock.now() - started) / duration));
    if (count !== lastCount) {
      lastCount = count;
      paint(count === segments.length ? text : segments.slice(0, count).join(''));
    }
    if (count < segments.length) frame = clock.request(tick);
    else frame = null;
  };
  frame = clock.request(tick);
  return () => {
    cancelled = true;
    if (frame !== null) clock.cancel(frame);
  };
}
