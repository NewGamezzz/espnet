<script setup>
import { computed } from "vue";
import { usePageData } from "vuepress/client";

const page = usePageData();

const editLink = computed(() => {
  const gitData = page.value.git;

  return gitData?.editLink ?? "";
});

const askChatGPT = () => {
  const url = window.location.href;
  const prompt = encodeURIComponent(
    `Read from ${url} so I can ask questions about its contents`,
  );

  window.open(`https://chatgpt.com/?hint=search&q=${prompt}`, "_blank");
};

const copyMarkdown = async () => {
  const title = page.value.title || document.title;
  const url = window.location.href;

  await navigator.clipboard.writeText(`# ${title}\n\nSource: ${url}\n`);
};
</script>

<template>
  <aside class="docs-ai-sidebar" aria-label="AI tools">
    <div class="docs-ai-sidebar__title">AI Tools</div>

    <div class="docs-ai-sidebar__actions">
      <button class="docs-ai-sidebar__action" type="button" @click="copyMarkdown">
        <svg
          class="docs-ai-sidebar__icon"
          viewBox="0 0 24 24"
          aria-hidden="true"
        >
          <path
            d="M6 4.75A2.75 2.75 0 0 1 8.75 2h4.69c.73 0 1.43.29 1.94.81l2.81 2.81c.52.51.81 1.21.81 1.94v11.69A2.75 2.75 0 0 1 16.25 22h-7.5A2.75 2.75 0 0 1 6 19.25zm8 .75v2.25c0 .69.56 1.25 1.25 1.25h2.25zM9 12.25c0-.41.34-.75.75-.75h4.5a.75.75 0 0 1 0 1.5h-4.5a.75.75 0 0 1-.75-.75m0 3c0-.41.34-.75.75-.75h4.5a.75.75 0 0 1 0 1.5h-4.5a.75.75 0 0 1-.75-.75"
            fill="currentColor"
          />
        </svg>
        Copy as Markdown
      </button>

      <button class="docs-ai-sidebar__action" type="button" @click="askChatGPT">
        <svg
          class="docs-ai-sidebar__icon"
          viewBox="0 0 24 24"
          aria-hidden="true"
        >
          <path
            d="M11.3 2.15a4.7 4.7 0 0 1 6.02 2.85 4.72 4.72 0 0 1 3.2 6.04 4.72 4.72 0 0 1-1.62 8.18 4.72 4.72 0 0 1-7.58 2.63 4.72 4.72 0 0 1-7.82-4.74 4.72 4.72 0 0 1-.02-8.26A4.72 4.72 0 0 1 8.3 2.7a4.7 4.7 0 0 1 3-.55m-1 2.03A2.96 2.96 0 0 0 8.1 5.5l-.42 1.01 2.8 1.62 2.2-1.27V5.61a2.96 2.96 0 0 0-2.38-1.43m4.14 3.4-2.2 1.27v3.28l2.84 1.64 1.09-.63a2.97 2.97 0 0 0 1.1-4.05 2.97 2.97 0 0 0-2.86-1.5zm-7.7 1.06A2.97 2.97 0 0 0 5.12 10c-.5.86-.54 1.9-.12 2.8l.5 1 2.83-1.64v-3.3L7.24 8.2zm1.6 5.16-2.82 1.63v1.26a2.97 2.97 0 0 0 4.45 2.57l.88-.5v-3.27zm3.89 0v3.27l.89.5a2.97 2.97 0 0 0 4.44-2.57v-1.26l-2.82-1.63zm-1.13-2.72-2.84 1.64v3.28l2.84 1.64 2.84-1.64v-3.28z"
            fill="currentColor"
          />
        </svg>
        Ask ChatGPT
      </button>

      <a class="docs-ai-sidebar__action" href="">
        <svg
          class="docs-ai-sidebar__icon"
          viewBox="0 0 24 24"
          aria-hidden="true"
        >
          <path
            d="M12 2 4.5 6.25v8.5L12 19l7.5-4.25v-8.5zm0 1.72 5.84 3.31L12 10.34 6.16 7.03zm-6 4.6 5.25 2.98v6.02L6 14.34zm12 0v6.02l-5.25 2.98V11.3z"
            fill="currentColor"
          />
        </svg>
        Ask Claude
      </a>

      <a class="docs-ai-sidebar__action" href="">
        <svg
          class="docs-ai-sidebar__icon"
          viewBox="0 0 24 24"
          aria-hidden="true"
        >
          <path
            d="M12 3.5a8.5 8.5 0 1 0 8.07 11.2h-3.2a5.5 5.5 0 1 1 0-5.4h3.2A8.5 8.5 0 0 0 12 3.5m0 3a5.5 5.5 0 0 1 4.73 2.7H12v5.5h5.04A5.5 5.5 0 1 1 12 6.5"
            fill="currentColor"
          />
        </svg>
        Ask Gemini
      </a>

      <a
        v-if="editLink"
        class="docs-ai-sidebar__action"
        :href="editLink"
        target="_blank"
        rel="noreferrer"
      >
        <svg
          class="docs-ai-sidebar__icon"
          viewBox="0 0 24 24"
          aria-hidden="true"
        >
          <path
            d="M12 .5C5.65.5.5 5.65.5 12A11.5 11.5 0 0 0 8.36 22.1c.58.11.79-.25.79-.56v-2.02c-2.38.52-2.88-1.01-2.88-1.01-.39-.97-.96-1.22-.96-1.22-.78-.53.06-.52.06-.52.87.06 1.32.9 1.32.9.76 1.31 2  .93 2.49.71.08-.56.3-.94.55-1.15-1.9-.22-3.89-.95-3.89-4.24 0-.94.34-1.71.89-2.31-.09-.22-.39-1.12.09-2.33 0 0 .73-.23 2.4.88a8.3 8.3 0 0 1 4.36 0c1.66-1.11 2.39-.88 2.39-.88.49 1.21.19 2.11.1 2.33.56.6.89 1.37.89 2.31 0 3.3-2 4.01-3.91 4.23.31.27.58.79.58 1.59v2.36c0 .31.2.68.8.56A11.5 11.5 0 0 0 23.5 12C23.5 5.65 18.35.5 12 .5"
            fill="currentColor"
          />
        </svg>
        Edit on GitHub
      </a>
    </div>
  </aside>
</template>
