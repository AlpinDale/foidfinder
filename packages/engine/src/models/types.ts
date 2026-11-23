import type { TagIndex } from "../tagging";
import type { DeviceFeatures } from "../webgpu/device";
import type { OnProgress } from "../webgpu/weights";

/** Minimum config.json shape shared by all tagger models. */
export interface ModelConfigBase {
  model: {
    name?: string;
    architecture: string;
  };
  tags: string[];
  categories: string[];
  thresholds: Record<string, number>;
}

/** Identifies where to load a model checkpoint from. */
export interface ModelSource {
  /** Base URL serving config.json and weight files. */
  baseUrl?: string;
  /** Optional registered model architecture id; auto-detected from config.json when omitted. */
  modelId?: string;
}

/** Runtime context supplied by the engine when instantiating a model on WebGPU. */
export interface ModelLoadContext {
  device: GPUDevice;
  features: DeviceFeatures;
  baseUrl: string;
  fetchFile: (url: string, onProgress?: OnProgress) => Promise<ArrayBuffer>;
  onProgress?: OnProgress;
}

/** A GPU-resident tagger model ready to score decoded ImageBitmaps. */
export interface LoadedModel {
  readonly id: string;
  readonly name: string;
  readonly tagCount: number;
  readonly index: TagIndex;
  /** Preprocesses and scores a decoded image -> float32 probabilities of length `tagCount`. */
  run(image: ImageBitmap): Promise<Float32Array>;
  /** Releases GPU resources owned by this model instance. */
  destroy(): void;
}

/** Pluggable architecture adapter for registering new tagger models with the engine. */
export interface ModelDefinition<TConfig extends ModelConfigBase = ModelConfigBase> {
  /** Unique architecture identifier (e.g. "haruka-eva02-tree"). */
  readonly id: string;
  /** Human-readable display name. */
  readonly name: string;
  /** Default checkpoint base URL (pinned commit / release). */
  readonly defaultBaseUrl: string;
  /** Validates whether a parsed config.json belongs to this model architecture. */
  matchesConfig(config: unknown): config is TConfig;
  /** Downloads/prepares weights and builds the GPU model instance. */
  load(ctx: ModelLoadContext, config: TConfig): Promise<LoadedModel>;
}

export interface ModelSummary {
  id: string;
  name: string;
  defaultBaseUrl: string;
}
