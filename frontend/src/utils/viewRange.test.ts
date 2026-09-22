import { describe, expect, it } from 'vitest';
import {
  DEFAULT_VIEW_RANGE,
  DEFAULT_VIEWPORT_RANGE,
  MIN_VIEW_SPAN,
  constrainViewRange,
  normalizeViewRange,
  displayLimitForRange,
  orthographicFrustumSize,
  RANGE_LIMIT,
  zoomViewRange
} from './viewRange';

describe('default computation and camera ranges', () => {
  it('keeps the larger computation domain while framing the central orbit region', () => {
    expect(DEFAULT_VIEW_RANGE).toEqual({ xMin: -5, xMax: 5, yMin: -9, yMax: 9 });
    expect(DEFAULT_VIEWPORT_RANGE).toEqual({ xMin: -3, xMax: 3, yMin: -3, yMax: 3 });
    expect(constrainViewRange(DEFAULT_VIEWPORT_RANGE, DEFAULT_VIEW_RANGE)).toEqual(DEFAULT_VIEWPORT_RANGE);
    expect(constrainViewRange(
      zoomViewRange(DEFAULT_VIEWPORT_RANGE, 3), DEFAULT_VIEW_RANGE,
    )).toEqual(DEFAULT_VIEW_RANGE);
  });

  it('fits both axes on narrow and wide viewports without distorting coordinates', () => {
    for (const [width, height] of [[1600, 900], [600, 900]]) {
      const frustum = orthographicFrustumSize(DEFAULT_VIEW_RANGE, width, height);
      expect(frustum.width).toBeGreaterThan(10);
      expect(frustum.height).toBeGreaterThan(18);
      expect(frustum.width / frustum.height).toBeCloseTo(width / height);
    }
  });
});

describe('normalizeViewRange', () => {
  it('orders and clamps values', () => {
    const result = normalizeViewRange({ xMin: 12, xMax: -5, yMin: -20, yMax: 3 });
    expect(result.xMin).toBe(-5);
    expect(result.xMax).toBe(RANGE_LIMIT);
    expect(result.yMin).toBe(-RANGE_LIMIT);
    expect(result.yMax).toBe(3);
  });

  it('expands zero-width ranges', () => {
    const result = normalizeViewRange({ xMin: 1, xMax: 1, yMin: 2, yMax: 2 });
    expect(result.xMax).toBeGreaterThan(result.xMin);
    expect(result.yMax).toBeGreaterThan(result.yMin);
  });
});

describe('constrainViewRange', () => {
  it('shifts a zoomed camera inside the computation domain without changing its span', () => {
    expect(constrainViewRange(
      { xMin: 1, xMax: 4, yMin: -4, yMax: -1 },
      { xMin: -2, xMax: 2, yMin: -2, yMax: 2 },
    )).toEqual({ xMin: -1, xMax: 2, yMin: -2, yMax: 1 });
  });

  it('uses the whole computation domain when the requested camera is larger', () => {
    const domain = { xMin: -2, xMax: 2, yMin: -1, yMax: 1 };
    expect(constrainViewRange(
      { xMin: -8, xMax: 8, yMin: -8, yMax: 8 },
      domain,
    )).toEqual(domain);
  });
});

describe('displayLimitForRange', () => {
  it('preserves the guarded default for ordinary views and expands for finite display-only ranges', () => {
    expect(displayLimitForRange({ xMin: -2, xMax: 2, yMin: -1, yMax: 1 })).toBe(RANGE_LIMIT);
    expect(displayLimitForRange({ xMin: -14, xMax: 18, yMin: -2, yMax: 12 })).toBe(18);
  });
});

describe('zoomViewRange', () => {
  it('zooms around the current center', () => {
    const result = zoomViewRange({ xMin: -2, xMax: 2, yMin: -1.5, yMax: 1.5 }, 0.8);

    expect(result.xMin).toBeCloseTo(-1.6);
    expect(result.xMax).toBeCloseTo(1.6);
    expect(result.yMin).toBeCloseTo(-1.2);
    expect(result.yMax).toBeCloseTo(1.2);
  });

  it('keeps zoom-out ranges inside the global bounds without shrinking their span', () => {
    const result = zoomViewRange({ xMin: 8, xMax: 10, yMin: -10, yMax: -8 }, 2);

    expect(result).toEqual({ xMin: 6, xMax: 10, yMin: -10, yMax: -6 });
  });

  it('stops zoom-in before the range becomes numerically unusable', () => {
    const result = zoomViewRange({ xMin: -0.01, xMax: 0.01, yMin: -0.01, yMax: 0.01 }, 0.1);

    expect(result.xMax - result.xMin).toBeCloseTo(MIN_VIEW_SPAN);
    expect(result.yMax - result.yMin).toBeCloseTo(MIN_VIEW_SPAN);
  });
});
