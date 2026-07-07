// Types for the browser / Cloudflare Workers entrypoints
// (`pancake-wasm/web`). These runtimes expose only the portable API — the
// Node-only file helpers (loadJsonFile / loadSnapshotFile) are not available
// here and are absent from this type so calling them is a compile error rather
// than a runtime throw.
export * from './pancake.js';

import type { PancakeApi } from './pancake.js';

declare const Pancake: PancakeApi;

export default Pancake;
