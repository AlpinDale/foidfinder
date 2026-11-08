import { describe, expect, it } from "vitest";

import { percent, splitTag, tagsForCopy } from "./tags";
import type { TagResult } from "./tagger";

describe("splitTag", () => {
  it("splits the series qualifier and title-cases both parts", () => {
    expect(splitTag("kafka_(honkai:_star_rail)")).toEqual({ name: "Kafka", qualifier: "Honkai: Star Rail" });
  });
  it("keeps names without a qualifier whole", () => {
    expect(splitTag("hatsune_miku")).toEqual({ name: "Hatsune Miku", qualifier: "" });
  });
  it("only treats a trailing parenthesis group as the qualifier", () => {
    expect(splitTag("ranma-chan_(ranma_1/2)")).toEqual({ name: "Ranma-Chan", qualifier: "Ranma 1/2" });
    expect(splitTag("(lol)_face")).toEqual({ name: "(lol) Face", qualifier: "" });
  });
});

describe("percent", () => {
  it("never rounds a non-certain score up to 100% or down to 0%", () => {
    expect(percent(0.9999)).toBe("99%");
    expect(percent(0.004)).toBe("<1%");
    expect(percent(0.274)).toBe("27%");
  });
});

it("tagsForCopy lists characters, then series, then general tags with spaces", () => {
  const t = (tag: string) => ({ tag, score: 0.9 });
  const result = {
    tags: { character: [t("nanasaki_ai")], copyright: [t("amagami")], general: [t("1girl"), t("school_uniform")], style: [t("sketch")], meta: [] },
  } as unknown as TagResult;
  expect(tagsForCopy(result)).toBe("nanasaki ai, amagami, 1girl, school uniform");
});
