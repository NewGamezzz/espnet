<template>
  <section class="home-doclinks">
    <div class="home-section-inner">
      <div class="dl-grid">
        <div v-for="col in columns" :key="col.heading" class="dl-col">
          <div class="dl-heading">{{ col.heading }}</div>
          <ul class="dl-list">
            <li v-for="link in col.links" :key="link.label">
              <a :href="link.href" class="dl-link">{{ link.label }}</a>
            </li>
          </ul>
        </div>
      </div>
    </div>
  </section>
</template>

<script setup>
import { withBase } from 'vuepress/client'

const columns = [
  {
    heading: 'Docs',
    links: [
      { label: 'Core Reference',    href: 'core/' },
      { label: 'Config Reference',  href: 'core/config/' },
      { label: 'System and Stages', href: 'core/system-and-stages' },
      { label: 'Components',        href: 'core/components/' },
      { label: 'Parallel & Distributed', href: 'core/parallel/' },
      { label: 'Stages Overview',   href: 'stages/' },
    ],
  },
  {
    heading: 'Tutorials',
    links: [
      { label: 'Quick Start',              href: 'get-started/index' },
      { label: 'What a Recipe Is',         href: 'get-started/what-is-a-recipe' },
      { label: 'Finetuning',               href: 'guides/finetuning/' },
      { label: 'Migrating from ESPnet2',   href: 'guides/migrating-from-espnet2/' },
      { label: 'Coming from PyTorch',      href: 'guides/coming-from-pytorch/' },
      { label: 'Coming from other tools',  href: 'guides/coming-from-other-toolkits/' },
      { label: 'Custom Dataset',           href: 'guides/finetuning/custom-dataset' },
      { label: 'Custom Model',             href: 'guides/finetuning/custom-model' },
    ],
  },
  {
    heading: 'Resources',
    links: [
      { label: 'GitHub',           href: 'https://github.com/espnet/espnet' },
      { label: 'Citation',         href: withBase('/citations.html') },
      { label: 'Contributing',     href: 'contributing/' },
      { label: 'Troubleshooting',  href: 'get-started/installation-troubleshooting' },
    ],
  },
]
</script>

<style scoped>
.dl-citation {
  margin-top: 2rem;
  padding-top: 1.5rem;
  border-top: 1px solid #e0e0e0;
}

.dl-citation-desc {
  font-size: 13px;
  color: var(--vp-c-text-2);
  margin: 8px 0 10px;
}

.dl-bibtex {
  background: var(--vp-c-bg-soft);
  border: 1px solid #e0e0e0;
  border-radius: 8px;
  padding: 14px 16px;
  font-size: 12px;
  line-height: 1.6;
  overflow-x: auto;
  margin: 0 0 10px;
}

.dl-bibtex code {
  font-family: var(--vp-font-family-mono);
  color: var(--vp-c-text-1);
  background: none;
  padding: 0;
}

.dl-citation-more {
  font-size: 12px;
  color: var(--vp-c-brand-2, #3ecf8e);
  text-decoration: none;
}

.dl-citation-more:hover {
  text-decoration: underline;
}

</style>
