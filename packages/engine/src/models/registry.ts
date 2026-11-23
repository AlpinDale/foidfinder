import type { DeviceFeatures } from "../webgpu/device";
import { fetchCached, type OnProgress } from "../webgpu/weights";
import { harukaModel } from "./haruka";
import type { LoadedModel, ModelConfigBase, ModelDefinition, ModelSource, ModelSummary } from "./types";

/**
 * Registry of supported WebGPU tagger model architectures.
 * New models register a `ModelDefinition` here without changing the worker, client, web UI, or extension.
 */
export class ModelRegistry {
  private models = new Map<string, ModelDefinition<ModelConfigBase>>();
  private fallbackId: string | null = null;

  register<TConfig extends ModelConfigBase>(definition: ModelDefinition<TConfig>): this {
    this.models.set(definition.id, definition as unknown as ModelDefinition<ModelConfigBase>);
    if (this.fallbackId === null) this.fallbackId = definition.id;
    return this;
  }

  get(id: string): ModelDefinition<ModelConfigBase> | undefined {
    return this.models.get(id);
  }

  defaultModel(): ModelDefinition<ModelConfigBase> {
    const def = this.fallbackId ? this.models.get(this.fallbackId) : undefined;
    if (!def) throw new Error("No models registered");
    return def;
  }

  list(): ModelSummary[] {
    return [...this.models.values()].map(({ id, name, defaultBaseUrl }) => ({ id, name, defaultBaseUrl }));
  }

  resolveConfig(config: unknown, modelId?: string): ModelDefinition<ModelConfigBase> {
    if (modelId) {
      const def = this.models.get(modelId);
      if (!def) throw new Error(`Unknown model architecture: ${modelId}`);
      if (!def.matchesConfig(config)) throw new Error(`Model config does not match architecture ${modelId}`);
      return def;
    }
    for (const def of this.models.values()) {
      if (def.matchesConfig(config)) return def;
    }
    const arch = (config as { model?: { architecture?: unknown } } | null)?.model?.architecture;
    throw new Error(`Unsupported model architecture${typeof arch === "string" ? `: ${arch}` : ""}`);
  }

  async load(
    device: GPUDevice,
    features: DeviceFeatures,
    source: ModelSource = {},
    onProgress?: OnProgress,
    fetchFile: (url: string, onProgress?: OnProgress) => Promise<ArrayBuffer> = fetchCached,
  ): Promise<LoadedModel> {
    const target = source.modelId ? this.get(source.modelId) : this.defaultModel();
    if (!target) throw new Error(`Unknown model architecture: ${source.modelId}`);
    const baseUrl = (source.baseUrl ?? target.defaultBaseUrl).replace(/\/+$/, "");
    const configBytes = await fetchFile(`${baseUrl}/config.json`);
    const config: unknown = JSON.parse(new TextDecoder().decode(configBytes));
    const definition = this.resolveConfig(config, source.modelId);
    return definition.load({ device, features, baseUrl, fetchFile, onProgress }, config as ModelConfigBase);
  }
}

export function createDefaultRegistry(): ModelRegistry {
  return new ModelRegistry().register(harukaModel);
}

export const defaultRegistry: ModelRegistry = /*#__PURE__*/ createDefaultRegistry();
