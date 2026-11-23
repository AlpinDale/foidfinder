import type { ModelConfigBase } from "../types";

export const HARUKA_ARCHITECTURE = "haruka-eva02-tree";
export const HARUKA_QUANT_FORMAT = "foidfinder-haruka-rowstep1";
export const HARUKA_MODEL_REPO = "alpindale/haruka-compressed";
/** Pinned commit so a cached copy in the Cache API always matches its URL. */
export const HARUKA_MODEL_REVISION = "c5fcbcb89b69a9a2453bc8825e3d8e0cb45dee4b";
export const HARUKA_DEFAULT_BASE = `https://huggingface.co/${HARUKA_MODEL_REPO}/resolve/${HARUKA_MODEL_REVISION}`;

export interface HarukaConfig extends ModelConfigBase {
  model: {
    name?: string;
    architecture: typeof HARUKA_ARCHITECTURE;
    img_size: number;
    patch_size: number;
    dim: number;
    heads: number;
    depth: number;
    mlp_hidden: number;
    pool_heads: number;
    coarse_queries: number;
    leaf_queries: number;
    leaf_dim: number;
    rope_ref_grid: number;
  };
  quantization: {
    format: string;
    layers: Record<string, { shape: [number, number]; planes: number }>;
  };
}

export function isHarukaConfig(value: unknown): value is HarukaConfig {
  if (!value || typeof value !== "object") return false;
  const cfg = value as Partial<HarukaConfig>;
  return (
    cfg.model?.architecture === HARUKA_ARCHITECTURE &&
    typeof cfg.quantization?.format === "string" &&
    Array.isArray(cfg.tags) &&
    Array.isArray(cfg.categories) &&
    typeof cfg.thresholds === "object" &&
    cfg.thresholds !== null
  );
}
