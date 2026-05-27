<script setup lang="ts">
import { hasGlobalComponent } from "@vuepress/helper/client";
import { computed } from "vue";
import { usePageData, usePageFrontmatter, withBase } from "vuepress/client";
import { RenderDefault } from "vuepress-shared/client";

import BreadCrumb from "@theme-hope/components/BreadCrumb";
import MarkdownContent from "@theme-hope/components/MarkdownContent";
import PageNav from "@theme-hope/components/PageNav";
import PageTitle from "@theme-hope/components/PageTitle";
import { useThemeLocaleData } from "@theme-hope/composables/index";
import PageMeta from "@theme-hope/modules/info/components/PageMeta";
import TOC from "@theme-hope/modules/info/components/TOC";
import { useDarkmode } from "@theme-hope/modules/outlook/composables/index";

import DocsAiSidebar from "../../components/DocsAiSidebar.vue";
import DocsFeedback from "../../components/DocsFeedback.vue";

const page = usePageData();
const frontmatter = usePageFrontmatter();
const { isDarkmode } = useDarkmode();
const themeLocale = useThemeLocaleData();

const tocEnable = computed(
  () => frontmatter.value.toc ?? true,
);
const headerDepth = computed(
  () => frontmatter.value.headerDepth ?? themeLocale.value.headerDepth ?? 2,
);
const hasTocItems = computed(
  () => (page.value.headers ?? []).length > 0,
);
</script>

<template>
  <main id="main-content" class="vp-page">
    <component :is="hasGlobalComponent('LocalEncrypt') ? 'LocalEncrypt' : RenderDefault">
      <div class="docs-page-shell">
        <div class="docs-page-shell__main">
          <div v-if="frontmatter.cover" class="page-cover">
            <img :src="withBase(frontmatter.cover)" alt="" no-view />
          </div>

          <BreadCrumb />
          <PageTitle />
          <MarkdownContent />
          <PageMeta />
          <PageNav />

          <component
            :is="'CommentService'"
            v-if="hasGlobalComponent('CommentService')"
            :darkmode="isDarkmode"
          />
        </div>

        <div class="docs-page-shell__sidebar">
          <DocsFeedback />
          <DocsAiSidebar />
          <div v-if="tocEnable && hasTocItems" class="docs-page-shell__toc">
            <div class="docs-page-shell__toc-title">On this page</div>
            <TOC :header-depth="headerDepth" />
          </div>
        </div>
      </div>
    </component>
  </main>
</template>
