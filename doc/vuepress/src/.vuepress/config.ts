import { defineUserConfig } from "vuepress";
import { viteBundler } from "@vuepress/bundler-vite";
import { path } from "vuepress/utils";

import theme from "./theme.js";

export default defineUserConfig({
  base: "/espnet_draft_home_page/",

  lang: "en-US",

  head: [
    [
      "script",
      {
        async: true,
        src: "https://www.googletagmanager.com/gtag/js?id=G-XTTNWSWN9Z",
      },
    ],
    [
      "script",
      {},
      `
        window.dataLayer = window.dataLayer || [];
        function gtag(){dataLayer.push(arguments);}
        gtag('js', new Date());
        gtag('config', 'G-XTTNWSWN9Z');
      `,
    ],
  ],

  bundler: viteBundler({
    viteOptions: {
      resolve: {
        alias: {
          "@theme-hope/components/NormalPage": path.resolve(
            __dirname,
            "./theme/components/NormalPage.vue",
          ),
        },
      },
      build: {
        sourcemap: false,
        rollupOptions: {
          output: {
            manualChunks() {
              return 'vendor';
            },
          },
        },
      },
    },
    vuePluginOptions: {},
  }),

  theme,

  // Enable it with pwa
  // shouldPrefetch: false,
});
