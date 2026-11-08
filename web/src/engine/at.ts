/** a[i], throwing when the index is out of range (the engine's indices are in range by construction). */
export function at<T>(a: ArrayLike<T>, i: number): T {
  const v = a[i];
  if (v === undefined) throw new RangeError(`index ${i} out of range (length ${a.length})`);
  return v;
}
