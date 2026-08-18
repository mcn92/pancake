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
        completeProfile: {
          enabled: true,
          vocab: '../create-pancake-search/src/inline-encoder/vocab.txt',
          weights: '../create-pancake-search/src/inline-encoder/encoder-weights.bin',
          model: 'sentence-transformers/all-MiniLM-L6-v2',
          maxTokens: 128,
        },
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
