import { at } from "./at";

export interface Coeffs {
  kk: Int32Array; // [out, ksize] fixed-point taps (22 fraction bits)
  bounds: Int32Array; // [out, 2] (first tap, tap count)
  ksize: number;
}

export interface ResamplePass {
  axis: 0 | 1;
  coeffs: Coeffs;
  outW: number;
  outH: number;
  region: [top: number, left: number, h: number, w: number];
  srcW: number;
  row0: number;
}

/** Pillow's precompute_coeffs + normalize_coeffs_8bpc for BICUBIC (a = -0.5), with the same float64 operations. */
export function bicubicCoeffs(inSize: number, outSize: number): Coeffs {
  const scale = inSize / outSize;
  const filterscale = Math.max(scale, 1.0);
  const support = 2.0 * filterscale;
  const ksize = Math.ceil(support) * 2 + 1;
  const kk = new Int32Array(outSize * ksize);
  const bounds = new Int32Array(outSize * 2);
  const w = new Float64Array(ksize);
  for (let i = 0; i < outSize; i++) {
    const center = (i + 0.5) * scale;
    const xmin = Math.max(Math.trunc(center - support + 0.5), 0);
    const xmax = Math.min(Math.trunc(center + support + 0.5), inSize) - xmin;
    let ww = 0.0;
    for (let t = 0; t < ksize; t++) {
      let v = 0.0;
      if (t < xmax) {
        const x = Math.abs((t + xmin - center + 0.5) * (1.0 / filterscale));
        v = x < 1.0 ? (1.5 * x - 2.5) * x * x + 1 : x < 2.0 ? (((x - 5) * x + 8) * x - 4) * -0.5 : 0.0;
      }
      w[t] = v;
      ww += v;
    }
    for (let t = 0; t < ksize; t++) {
      const v = ww !== 0.0 ? at(w, t) / ww : at(w, t);
      kk[i * ksize + t] = Math.trunc(v < 0 ? -0.5 + v * (1 << 22) : 0.5 + v * (1 << 22));
    }
    bounds[i * 2] = xmin;
    bounds[i * 2 + 1] = xmax;
  }
  return { kk, bounds, ksize };
}

function identity(n: number): Coeffs {
  const bounds = new Int32Array(n * 2);
  for (let i = 0; i < n; i++) {
    bounds[i * 2] = i;
    bounds[i * 2 + 1] = 1;
  }
  return { kk: new Int32Array(n).fill(1 << 22), bounds, ksize: 1 };
}

/** Plans separable Pillow-exact bicubic letterbox passes for a w x h image onto a size x size square. */
export function planResizePad(w: number, h: number, size: number): ResamplePass[] {
  if (w === size && h === size) {
    return [{ axis: 0, coeffs: identity(size), outW: size, outH: size, region: [0, 0, size, size], srcW: w, row0: 0 }];
  }
  const r = Math.min(size / w, size / h);
  const nw = Math.max(1, Math.trunc(w * r));
  const nh = Math.max(1, Math.trunc(h * r));
  const top = Math.trunc((size - nh) / 2);
  const left = Math.trunc((size - nw) / 2);
  const ch = bicubicCoeffs(w, nw);
  const cv = bicubicCoeffs(h, nh);
  if (h - nh > 0 && h - nh > (w - nw) * 2) {
    return [
      { axis: 1, coeffs: cv, outW: w, outH: nh, region: [0, 0, nh, w], srcW: w, row0: 0 },
      { axis: 0, coeffs: ch, outW: size, outH: size, region: [top, left, nh, nw], srcW: w, row0: 0 },
    ];
  }
  const row0 = at(cv.bounds, 0);
  const last = (nh - 1) * 2;
  const rows = at(cv.bounds, last) + at(cv.bounds, last + 1) - row0;
  const shifted = { ...cv, bounds: cv.bounds.map((v, i) => (i % 2 === 0 ? v - row0 : v)) };
  return [
    { axis: 0, coeffs: ch, outW: nw, outH: rows, region: [0, 0, rows, nw], srcW: w, row0 },
    { axis: 1, coeffs: shifted, outW: size, outH: size, region: [top, left, nh, nw], srcW: nw, row0: 0 },
  ];
}
