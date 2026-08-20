// The one-file reader now lives in the published package — this example
// imports it the way any consumer does:
//
//   import { openPancakeFile } from 'pancake-wasm/complete';
//
// (Relative here because the example lives inside the pancake repo itself.)
export * from '../../complete/index.mjs';
