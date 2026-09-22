import type { ViewRange } from '../types/domain';

export const RANGE_LIMIT = 10;
export const MIN_VIEW_SPAN = 0.05;
export const ZOOM_IN_FACTOR = 0.8;
export const ZOOM_OUT_FACTOR = 1.25;

export const DEFAULT_VIEW_RANGE: ViewRange = {
  xMin: -5,
  xMax: 5,
  yMin: -9,
  yMax: 9
};

/** Start near the usual fixed and periodic orbits without restricting computation. */
export const DEFAULT_VIEWPORT_RANGE: ViewRange = {
  xMin: -3,
  xMax: 3,
  yMin: -3,
  yMax: 3
};

export const orthographicFrustumSize = (
  range: ViewRange,
  viewportWidth: number,
  viewportHeight: number,
  padding = 0.12,
): { width: number; height: number } => {
  const aspect = viewportWidth / viewportHeight;
  const height = Math.max(
    range.yMax - range.yMin + padding * 2,
    (range.xMax - range.xMin + padding * 2) / aspect,
  );
  return { width: height * aspect, height };
};

const clamp = (value: number, min: number, max: number): number => Math.max(min, Math.min(max, value));
const isFiniteNumber = (value: unknown): value is number => (
  typeof value === 'number' && Number.isFinite(value)
);

export const displayLimitForRange = (range: Partial<ViewRange>, baseLimit = RANGE_LIMIT): number => Math.max(
  baseLimit,
  ...Object.values(range || {}).filter(isFiniteNumber).map(value => Math.abs(value))
);

export const normalizeViewRange = (range: Partial<ViewRange>, limit = RANGE_LIMIT): ViewRange => {
  const xMin = isFiniteNumber(range.xMin) ? range.xMin : DEFAULT_VIEW_RANGE.xMin;
  const xMax = isFiniteNumber(range.xMax) ? range.xMax : DEFAULT_VIEW_RANGE.xMax;
  const yMin = isFiniteNumber(range.yMin) ? range.yMin : DEFAULT_VIEW_RANGE.yMin;
  const yMax = isFiniteNumber(range.yMax) ? range.yMax : DEFAULT_VIEW_RANGE.yMax;

  let loX = Math.min(xMin, xMax);
  let hiX = Math.max(xMin, xMax);
  let loY = Math.min(yMin, yMax);
  let hiY = Math.max(yMin, yMax);

  loX = clamp(loX, -limit, limit);
  hiX = clamp(hiX, -limit, limit);
  loY = clamp(loY, -limit, limit);
  hiY = clamp(hiY, -limit, limit);

  if (Math.abs(hiX - loX) < 1e-6) {
    const center = (hiX + loX) / 2;
    loX = clamp(center - 1, -limit, limit);
    hiX = clamp(center + 1, -limit, limit);
  }

  if (Math.abs(hiY - loY) < 1e-6) {
    const center = (hiY + loY) / 2;
    loY = clamp(center - 1, -limit, limit);
    hiY = clamp(center + 1, -limit, limit);
  }

  return { xMin: loX, xMax: hiX, yMin: loY, yMax: hiY };
};

const constrainAxis = (
  min: number,
  max: number,
  domainMin: number,
  domainMax: number,
): [number, number] => {
  const domainSpan = domainMax - domainMin;
  const span = max - min;
  if (span >= domainSpan) return [domainMin, domainMax];
  let low = min;
  let high = max;
  if (low < domainMin) {
    high += domainMin - low;
    low = domainMin;
  }
  if (high > domainMax) {
    low -= high - domainMax;
    high = domainMax;
  }
  return [Math.max(domainMin, low), Math.min(domainMax, high)];
};

/** Keep a camera viewport compactly inside the canonical computation domain. */
export const constrainViewRange = (range: Partial<ViewRange>, domain: ViewRange): ViewRange => {
  const normalizedDomain = normalizeViewRange(domain);
  const normalized = normalizeViewRange(range);
  const [xMin, xMax] = constrainAxis(
    normalized.xMin,
    normalized.xMax,
    normalizedDomain.xMin,
    normalizedDomain.xMax,
  );
  const [yMin, yMax] = constrainAxis(
    normalized.yMin,
    normalized.yMax,
    normalizedDomain.yMin,
    normalizedDomain.yMax,
  );
  return { xMin, xMax, yMin, yMax };
};

const zoomAxis = (
  min: number,
  max: number,
  factor: number,
  limit: number,
  minSpan: number,
): [number, number] => {
  const center = (min + max) / 2;
  const span = clamp((max - min) * factor, minSpan, limit * 2);
  let low = center - span / 2;
  let high = center + span / 2;

  if (low < -limit) {
    high += -limit - low;
    low = -limit;
  }
  if (high > limit) {
    low -= high - limit;
    high = limit;
  }

  return [clamp(low, -limit, limit), clamp(high, -limit, limit)];
};

export const zoomViewRange = (
  range: Partial<ViewRange>,
  factor: number,
  limit = RANGE_LIMIT,
  minSpan = MIN_VIEW_SPAN
): ViewRange => {
  if (!Number.isFinite(factor) || factor <= 0) {
    return normalizeViewRange(range, limit);
  }

  const normalized = normalizeViewRange(range, limit);
  const [xMin, xMax] = zoomAxis(normalized.xMin, normalized.xMax, factor, limit, minSpan);
  const [yMin, yMax] = zoomAxis(normalized.yMin, normalized.yMax, factor, limit, minSpan);
  return { xMin, xMax, yMin, yMax };
};
