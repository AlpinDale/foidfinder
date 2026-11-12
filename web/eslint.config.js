import js from "@eslint/js";
import reactHooks from "eslint-plugin-react-hooks";
import globals from "globals";
import tseslint from "typescript-eslint";

export default tseslint.config(
  { ignores: ["node_modules", "dist-extension"] },
  js.configs.recommended,
  ...tseslint.configs.strict,
  reactHooks.configs.flat["recommended-latest"],
  { languageOptions: { globals: globals.browser } },
);
