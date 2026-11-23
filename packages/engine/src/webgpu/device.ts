/** WebGPU device negotiation and capability flags shared across all models. */

export interface DeviceFeatures {
  /** Subgroup ops usable for 16-lane row reductions (subgroups enabled and minimum subgroup size >= 16). */
  subgroups16: boolean;
}

export interface DeviceContext {
  device: GPUDevice;
  features: DeviceFeatures;
  description: string;
}

/** Requests a WebGPU adapter and device with shader-f16 and optional subgroups/timestamps. */
export async function requestDevice(): Promise<DeviceContext> {
  if (!("gpu" in navigator)) throw new Error("This browser has no WebGPU, which foidfinder needs to run the model");
  const adapter = await navigator.gpu.requestAdapter({ powerPreference: "high-performance" });
  if (!adapter) throw new Error("WebGPU is turned on, but the browser offers no GPU to it");
  if (!adapter.features.has("shader-f16")) {
    throw new Error("This GPU or browser lacks 16-bit floats in WebGPU shaders (shader-f16), which the model needs");
  }
  const info = adapter.info as GPUAdapterInfo & { subgroupMinSize?: number };
  const subgroups16 = adapter.features.has("subgroups") && (info.subgroupMinSize ?? 0) >= 16;
  const required: GPUFeatureName[] = ["shader-f16"];
  if (subgroups16) required.push("subgroups" as GPUFeatureName);
  if (adapter.features.has("timestamp-query")) required.push("timestamp-query");
  const device = await adapter.requestDevice({
    requiredFeatures: required,
    requiredLimits: {
      maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize,
      maxBufferSize: adapter.limits.maxBufferSize,
      maxComputeWorkgroupStorageSize: adapter.limits.maxComputeWorkgroupStorageSize,
      maxStorageBuffersPerShaderStage: Math.max(8, Math.min(10, adapter.limits.maxStorageBuffersPerShaderStage)),
      maxTextureDimension2D: adapter.limits.maxTextureDimension2D,
    },
  });
  const description = [info.vendor, info.architecture, info.description].filter(Boolean).join(" ") || "WebGPU";
  return { device, features: { subgroups16 }, description };
}
