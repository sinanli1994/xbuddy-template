'use client';

import { useEffect, useState, type ReactNode } from 'react';
import { startTextReveal } from '@/utils/textReveal';

/** Only newly received message-only replies animate; history/tokens never do. */
export default function ProgressiveText({ text, animate, onProgress, children }: {
  text: string;
  animate: boolean;
  onProgress: () => void;
  children: (visible: string) => ReactNode;
}) {
  const [visible, setVisible] = useState(() => animate ? '' : text);
  const [reducedMotion, setReducedMotion] = useState(false);

  useEffect(() => {
    if (!animate) return;
    const preference = window.matchMedia('(prefers-reduced-motion: reduce)');
    let cancel = () => {};
    const finish = () => {
      cancel();
      setVisible(text);
      setReducedMotion(true);
      onProgress();
    };
    if (preference.matches) finish();
    else cancel = startTextReveal(text, next => {
      setVisible(next);
      onProgress();
    }, {
      now: () => performance.now(), request: callback => requestAnimationFrame(callback),
      cancel: id => cancelAnimationFrame(id),
    });
    const preferenceChanged = () => { if (preference.matches) finish(); };
    preference.addEventListener('change', preferenceChanged);
    return () => {
      cancel();
      preference.removeEventListener('change', preferenceChanged);
    };
  }, [text, animate, onProgress]);

  return children(animate && !reducedMotion ? visible : text);
}
