import { createRequire } from 'node:module';

const require = createRequire(import.meta.url);

const config = {
  title: 'Pancake',
  tagline: 'One-file semantic search artifacts for edge and browser runtimes',
  url: 'https://mcn92.github.io',
  baseUrl: '/pancake/',
  organizationName: 'mcn92',
  projectName: 'pancake',
  trailingSlash: false,
  onBrokenLinks: 'throw',
  markdown: {
    hooks: {
      onBrokenMarkdownLinks: 'warn',
    },
  },

  presets: [
    [
      'classic',
      {
        docs: {
          sidebarPath: './sidebars.js',
          editUrl: 'https://github.com/mcn92/pancake/tree/main/docs-site/',
        },
        blog: false,
        theme: {
          customCss: './src/css/custom.css',
        },
      },
    ],
  ],

  plugins: [
    [
      require.resolve('create-pancake-search/docusaurus'),
      {
        assetBase: 'pancake-search',
        sourcePath: 'docs',
        sourceRouteBase: 'docs',
        // No completeProfile block: the complete kind-3 .pancake is the
        // plugin default as of 0.7.0, staging the packaged encoder assets
        // (weights digest-pinned, fetched once) — this site exercises the
        // zero-config path.
        chunking: {
          targetTokens: 256,
          overlapPercent: 15,
        },
        index: {
          M: 12,
          efConstruction: 75,
          efSearch: 120,
        },
      },
    ],
  ],

  themeConfig: {
    navbar: {
      title: 'Pancake',
      items: [
        { type: 'docSidebar', sidebarId: 'docs', position: 'left', label: 'Docs' },
        { href: 'https://github.com/mcn92/pancake', label: 'GitHub', position: 'right' },
      ],
    },
    footer: {
      style: 'dark',
      links: [
        {
          title: 'Docs',
          items: [
            { label: 'Introduction', to: '/docs/intro' },
            { label: 'Quickstart', to: '/docs/guides/quickstart' },
            { label: 'Formats', to: '/docs/reference/formats' },
          ],
        },
      ],
      copyright: `Copyright ${new Date().getFullYear()} Pancake contributors.`,
    },
  },
};

export default config;
