import { describe, expect, it } from "vitest";

import { TagIndex } from "../tagging";
import { HARUKA_ARCHITECTURE } from "./haruka";
import { defaultRegistry, ModelRegistry } from "./registry";
import type { LoadedModel, ModelConfigBase, ModelDefinition } from "./types";

interface DummyConfig extends ModelConfigBase {
  model: { name: string; architecture: "dummy-vit" };
}

const dummyModel: ModelDefinition<DummyConfig> = {
  id: "dummy-vit",
  name: "Dummy Tagger",
  defaultBaseUrl: "https://example.test/dummy",
  matchesConfig(cfg): cfg is DummyConfig {
    return (cfg as Partial<DummyConfig>)?.model?.architecture === "dummy-vit";
  },
  async load(_ctx, config): Promise<LoadedModel> {
    return {
      id: "dummy-vit",
      name: config.model.name,
      tagCount: config.tags.length,
      index: new TagIndex(config.tags, config.categories, config.thresholds),
      async run() {
        return new Float32Array(config.tags.length);
      },
      destroy() {},
    };
  },
};

describe("ModelRegistry", () => {
  it("registers Haruka as the default built-in model", () => {
    expect(defaultRegistry.defaultModel().id).toBe(HARUKA_ARCHITECTURE);
    expect(defaultRegistry.list().map((m) => m.id)).toEqual([HARUKA_ARCHITECTURE]);
  });

  it("resolves and loads pluggable model architectures from config.json", async () => {
    const registry = new ModelRegistry().register(dummyModel);
    const config: DummyConfig = {
      model: { name: "Dummy Tagger", architecture: "dummy-vit" },
      tags: ["1girl"],
      categories: ["general"],
      thresholds: { general: 0.35 },
    };
    const fetchFile = async (url: string) => {
      expect(url).toBe("https://example.test/dummy/config.json");
      return new TextEncoder().encode(JSON.stringify(config)).buffer;
    };
    const loaded = await registry.load({} as GPUDevice, { subgroups16: false }, {}, undefined, fetchFile);
    expect(loaded.id).toBe("dummy-vit");
    expect(loaded.name).toBe("Dummy Tagger");
    expect(loaded.tagCount).toBe(1);
  });

  it("throws a clear error on unsupported architectures", () => {
    const registry = new ModelRegistry().register(dummyModel);
    expect(() => registry.resolveConfig({ model: { architecture: "unknown-arch" } })).toThrow("Unsupported model architecture: unknown-arch");
  });
});
