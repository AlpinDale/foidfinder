import type { ModelDefinition } from "../types";
import { HARUKA_ARCHITECTURE, HARUKA_DEFAULT_BASE, isHarukaConfig, type HarukaConfig } from "./config";
import { HarukaModel } from "./model";

export { HARUKA_ARCHITECTURE, HARUKA_DEFAULT_BASE, HARUKA_MODEL_REPO, HARUKA_MODEL_REVISION, isHarukaConfig, type HarukaConfig } from "./config";
export { HarukaModel } from "./model";

export const harukaModel: ModelDefinition<HarukaConfig> = {
  id: HARUKA_ARCHITECTURE,
  name: "Haruka Tagger",
  defaultBaseUrl: HARUKA_DEFAULT_BASE,
  matchesConfig: isHarukaConfig,
  async load(ctx, config) {
    const weights = await ctx.fetchFile(`${ctx.baseUrl}/model.safetensors`, ctx.onProgress);
    return HarukaModel.load(ctx.device, ctx.features, config, weights);
  },
};
