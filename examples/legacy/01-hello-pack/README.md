# Hello Pack

Minimal bundled browser consumer for `pikelet-wasm/web`. This is the fixture
behind `npm run test:browser`: it installs the local package tarball, starts
Vite, loads the web entrypoint in Playwright, and checks that a packaged index
can be opened and searched in Chromium.

From the repository root:

```bash
npm run test:browser
```
