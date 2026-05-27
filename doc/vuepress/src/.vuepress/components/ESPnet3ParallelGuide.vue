<script setup>
import SectionConfigBuilder from "./SectionConfigBuilder.vue";
import SectionExecutionModes from "./SectionExecutionModes.vue";
import SectionOverview from "./SectionOverview.vue";
import SectionProvider from "./SectionProvider.vue";
import SectionShardLifecycle from "./SectionShardLifecycle.vue";
</script>

<template>
  <div class="espnet3-guide">
    <SectionOverview />
    <SectionExecutionModes />
    <SectionProvider />
    <SectionShardLifecycle />
    <SectionConfigBuilder />
  </div>
</template>

<style>
.espnet3-guide {
  --bg: #ffffff;
  --bg2: #f8f8fb;
  --bg3: #f0f0f6;
  --border: rgba(0, 0, 0, 0.08);
  --border2: rgba(0, 0, 0, 0.14);
  --text: #1a1a2e;
  --text2: #5a5a7a;
  --text3: #9898b0;
  --driver: #2563eb;
  --driver-bg: rgba(37, 99, 235, 0.07);
  --driver-border: rgba(37, 99, 235, 0.35);
  --worker: #0d9e6a;
  --worker-bg: rgba(13, 158, 106, 0.07);
  --worker-border: rgba(13, 158, 106, 0.35);
  --provider: #b45309;
  --provider-bg: rgba(180, 83, 9, 0.07);
  --provider-border: rgba(180, 83, 9, 0.35);
  --forward: #7c3aed;
  --forward-bg: rgba(124, 58, 237, 0.07);
  --forward-border: rgba(124, 58, 237, 0.35);
  --done: #16a34a;
  --done-bg: rgba(22, 163, 74, 0.07);
  --done-border: rgba(22, 163, 74, 0.35);
  --mono: var(--font-family-code, "JetBrains Mono", monospace);
  --sans: var(--font-family, "Sora", sans-serif);

  background: var(--bg);
  color: var(--text);
  font-family: var(--sans);
  font-size: 15px;
  line-height: 1.7;
}

.espnet3-guide,
.espnet3-guide * {
  box-sizing: border-box;
}

.espnet3-guide .guide-section {
  padding: 72px 0;
  border-bottom: 1px solid var(--border);
}

.espnet3-guide .guide-section:last-child {
  border-bottom: none;
}

.espnet3-guide .container {
  max-width: 900px;
  margin: 0 auto;
  padding: 0 2rem;
}

.espnet3-guide .section-num,
.espnet3-guide .tab-btn,
.espnet3-guide .exec-col-title,
.espnet3-guide .shard-box-title,
.espnet3-guide .shard-items,
.espnet3-guide .config-label,
.espnet3-guide .config-section-label,
.espnet3-guide .role-label,
.espnet3-guide .method-tag,
.espnet3-guide .legend-item {
  font-family: var(--mono);
}

.espnet3-guide .section-num {
  margin-bottom: 0.5rem;
  color: var(--text3);
  font-size: 12px;
  letter-spacing: 0.1em;
}

.espnet3-guide .section-h2 {
  margin-bottom: 0.75rem;
  font-size: 26px;
  font-weight: 600;
}

.espnet3-guide .section-desc {
  max-width: 620px;
  margin-bottom: 2.5rem;
  color: var(--text2);
  font-size: 15px;
  line-height: 1.75;
}

.espnet3-guide pre,
.espnet3-guide code {
  font-family: var(--mono);
  font-size: 13px;
}

.espnet3-guide pre {
  overflow-x: auto;
  padding: 0.85rem 1rem;
  border: 1px solid var(--border);
  border-radius: 8px;
  background: var(--bg3);
  line-height: 1.7;
}

.espnet3-guide pre > code,
.espnet3-guide .closure-box code,
.espnet3-guide .inject-demo code {
  display: block;
  padding: 0;
  border: 0;
  border-radius: 0;
  background: transparent;
  color: inherit;
  white-space: pre;
}

.espnet3-guide .closure-box code,
.espnet3-guide .inject-demo code {
  white-space: pre-wrap;
}

.espnet3-guide .kw {
  color: var(--forward);
}

.espnet3-guide .fn {
  color: var(--driver);
}

.espnet3-guide .str {
  color: var(--worker);
}

.espnet3-guide .cm {
  color: var(--text3);
}

.espnet3-guide .num {
  color: var(--provider);
}

.espnet3-guide .tab-row,
.espnet3-guide .legend,
.espnet3-guide .role-methods,
.espnet3-guide .step-indicators,
.espnet3-guide .stepper-nav,
.espnet3-guide .indices-strip,
.espnet3-guide .shards-row {
  display: flex;
  flex-wrap: wrap;
  gap: 0.5rem;
}

.espnet3-guide .tab-row {
  margin-bottom: 1.5rem;
}

.espnet3-guide .tab-btn,
.espnet3-guide .btn {
  cursor: pointer;
  border: 1px solid var(--border2);
  border-radius: 6px;
  background: transparent;
  color: var(--text2);
  font-size: 12px;
  transition: all 0.18s;
}

.espnet3-guide .tab-btn {
  padding: 6px 14px;
}

.espnet3-guide .btn {
  padding: 0.55rem 0.9rem;
}

.espnet3-guide .tab-btn:hover,
.espnet3-guide .btn:hover {
  border-color: rgba(0, 0, 0, 0.2);
  background: var(--bg3);
  color: var(--text);
}

.espnet3-guide .tab-btn.active,
.espnet3-guide .btn-primary {
  border-color: var(--driver-border);
  background: var(--driver-bg);
  color: var(--driver);
}

.espnet3-guide .btn:disabled {
  cursor: not-allowed;
  opacity: 0.35;
}

.espnet3-guide .exec-diagram {
  display: flex;
  align-items: stretch;
  overflow: hidden;
  margin-bottom: 1.5rem;
  border: 1px solid var(--border);
  border-radius: 12px;
}

.espnet3-guide .exec-col {
  flex: 1;
  padding: 1.25rem;
  border-right: 1px solid var(--border);
}

.espnet3-guide .exec-col:last-child {
  border-right: none;
}

.espnet3-guide .exec-col-title {
  margin-bottom: 1rem;
  font-size: 11px;
  letter-spacing: 0.06em;
  text-transform: uppercase;
}

.espnet3-guide .col-driver {
  background: var(--driver-bg);
}

.espnet3-guide .col-worker {
  background: var(--worker-bg);
}

.espnet3-guide .exec-box,
.espnet3-guide .shard-box,
.espnet3-guide .role-card,
.espnet3-guide .provider-half,
.espnet3-guide .config-controls,
.espnet3-guide .config-output,
.espnet3-guide .shard-visual {
  border: 1px solid var(--border);
  border-radius: 8px;
  background: var(--bg2);
}

.espnet3-guide .exec-box {
  margin-bottom: 0.5rem;
  padding: 0.75rem 1rem;
  background: var(--bg3);
  font-size: 13px;
}

.espnet3-guide .exec-box-title {
  margin-bottom: 0.25rem;
  font-weight: 500;
}

.espnet3-guide .exec-box-sub,
.espnet3-guide .stage-desc,
.espnet3-guide .role-card p,
.espnet3-guide .step-line,
.espnet3-guide .method-tag,
.espnet3-guide .legend-item,
.espnet3-guide .env-note {
  color: var(--text2);
}

.espnet3-guide .exec-box-sub {
  font-size: 12px;
}

.espnet3-guide .exec-step-arrow,
.espnet3-guide .exec-arrow {
  color: var(--text3);
  text-align: center;
}

.espnet3-guide .exec-step-arrow {
  margin: 0.3rem 0;
  font-size: 13px;
}

.espnet3-guide .exec-arrow {
  display: flex;
  align-items: center;
  justify-content: center;
  padding: 0 0.5rem;
  font-size: 18px;
}

.espnet3-guide .box-driver,
.espnet3-guide .shard-box.pending {
  border-color: var(--driver-border);
  background: var(--driver-bg);
}

.espnet3-guide .box-worker {
  border-color: var(--worker-border);
  background: var(--worker-bg);
}

.espnet3-guide .box-provider,
.espnet3-guide .shard-box.running {
  border-color: var(--provider-border);
  background: var(--provider-bg);
}

.espnet3-guide .box-forward {
  border-color: var(--forward-border);
  background: var(--forward-bg);
}

.espnet3-guide .box-done,
.espnet3-guide .shard-box.complete {
  border-color: var(--done-border);
  background: var(--done-bg);
}

.espnet3-guide .box-driver .exec-box-title,
.espnet3-guide .shard-box.pending .shard-box-title,
.espnet3-guide .role-card.c-driver .role-label {
  color: var(--driver);
}

.espnet3-guide .box-worker .exec-box-title,
.espnet3-guide .role-card.c-worker .role-label {
  color: var(--worker);
}

.espnet3-guide .box-provider .exec-box-title,
.espnet3-guide .shard-box.running .shard-box-title,
.espnet3-guide .role-card.c-provider .role-label {
  color: var(--provider);
}

.espnet3-guide .box-forward .exec-box-title,
.espnet3-guide .role-card.c-forward .role-label {
  color: var(--forward);
}

.espnet3-guide .box-done .exec-box-title,
.espnet3-guide .shard-box.complete .shard-box-title {
  color: var(--done);
}

.espnet3-guide .callout {
  display: flex;
  gap: 0.75rem;
  margin: 1.25rem 0;
  padding: 1rem 1.25rem;
  border-radius: 8px;
}

.espnet3-guide .callout-info {
  border: 1px solid rgba(37, 99, 235, 0.2);
  background: rgba(37, 99, 235, 0.06);
}

.espnet3-guide .callout-warn {
  border: 1px solid rgba(180, 83, 9, 0.2);
  background: rgba(180, 83, 9, 0.06);
}

.espnet3-guide .callout-tip {
  border: 1px solid rgba(13, 158, 106, 0.2);
  background: rgba(13, 158, 106, 0.06);
}

.espnet3-guide .callout-icon {
  flex-shrink: 0;
  margin-top: 1px;
  font-size: 16px;
}

.espnet3-guide .callout-info .callout-icon,
.espnet3-guide .callout a {
  color: var(--driver);
}

.espnet3-guide .callout-warn .callout-icon {
  color: var(--provider);
}

.espnet3-guide .callout-tip .callout-icon {
  color: var(--worker);
}

.espnet3-guide .callout p {
  color: var(--text2);
}

.espnet3-guide .callout strong {
  color: var(--text);
  font-weight: 500;
}

.espnet3-guide .callout code {
  padding: 1px 5px;
  border-radius: 3px;
  background: var(--bg3);
  color: var(--driver);
  font-size: 12px;
}

.espnet3-guide .shards-row {
  gap: 1rem;
  margin: 1rem 0;
}

.espnet3-guide .shard-box {
  flex: 1;
  min-width: 130px;
  padding: 0.75rem 1rem;
  background: var(--bg3);
  font-size: 12px;
}

.espnet3-guide .shard-box-title,
.espnet3-guide .shard-status,
.espnet3-guide .closure-title,
.espnet3-guide .timing-badge,
.espnet3-guide .config-output-header,
.espnet3-guide .sbatch-title,
.espnet3-guide .step-dot,
.espnet3-guide .step-ind {
  font-family: var(--mono);
  font-size: 11px;
}

.espnet3-guide .shard-box-title {
  margin-bottom: 0.4rem;
}

.espnet3-guide .shard-status {
  margin-top: 0.5rem;
}

.espnet3-guide .dir-tree,
.espnet3-guide .closure-box,
.espnet3-guide .inject-demo,
.espnet3-guide .sbatch-preview {
  border: 1px solid var(--border2);
  border-radius: 8px;
  background: var(--bg3);
  color: var(--text2);
  font-family: var(--mono);
  font-size: 12px;
  line-height: 1.7;
}

.espnet3-guide .dir-tree,
.espnet3-guide .closure-box,
.espnet3-guide .inject-demo {
  padding: 0.75rem 1rem;
}

.espnet3-guide .dir-driver,
.espnet3-guide .inject-auto {
  color: var(--driver);
}

.espnet3-guide .dir-done {
  color: var(--done);
}

.espnet3-guide .dir-provider {
  color: var(--provider);
}

.espnet3-guide .inject-match {
  color: var(--worker);
}

.espnet3-guide .inject-match,
.espnet3-guide .inject-auto {
  padding: 1px 4px;
  border-radius: 3px;
}

.espnet3-guide .inject-match {
  background: var(--worker-bg);
}

.espnet3-guide .inject-auto {
  background: var(--driver-bg);
  font-style: italic;
}

.espnet3-guide .indices-strip {
  gap: 4px;
  margin-bottom: 1rem;
}

.espnet3-guide .idx-cell {
  min-width: 28px;
  padding: 3px 6px;
  border: 1px solid var(--border);
  border-radius: 4px;
  background: var(--bg3);
  font-family: var(--mono);
  font-size: 11px;
  text-align: center;
}

.espnet3-guide .idx-cell.shard-0 {
  border-color: var(--driver-border);
  background: var(--driver-bg);
  color: var(--driver);
}

.espnet3-guide .idx-cell.shard-1 {
  border-color: var(--worker-border);
  background: var(--worker-bg);
  color: var(--worker);
}

.espnet3-guide .idx-cell.shard-2 {
  border-color: var(--provider-border);
  background: var(--provider-bg);
  color: var(--provider);
}

.espnet3-guide .idx-cell.shard-3 {
  border-color: var(--forward-border);
  background: var(--forward-bg);
  color: var(--forward);
}

.espnet3-guide .shard-visual,
.espnet3-guide .config-controls,
.espnet3-guide .provider-half-body {
  padding: 1.25rem;
}

.espnet3-guide .env-note {
  padding: 0.75rem 1rem;
  border: 1px solid var(--border);
  border-radius: 8px;
  background: var(--bg2);
  font-size: 12px;
}

.espnet3-guide .env-note strong {
  color: var(--provider);
  font-weight: 500;
}

.espnet3-guide .config-grid,
.espnet3-guide .overview-grid {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 1.5rem;
  align-items: start;
}

.espnet3-guide .provider-split {
  display: grid;
  grid-template-columns: 1fr;
  gap: 1.5rem;
  margin-bottom: 2rem;
}

.espnet3-guide .config-output {
  position: sticky;
  top: 16px;
  overflow: hidden;
}

.espnet3-guide .config-output-header,
.espnet3-guide .provider-half-header,
.espnet3-guide .stepper-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
}

.espnet3-guide .config-output-header,
.espnet3-guide .provider-half-header {
  padding: 0.75rem 1.25rem;
  border-bottom: 1px solid var(--border);
  color: var(--text3);
}

.espnet3-guide .config-output pre {
  margin: 0;
  padding: 0.85rem 1rem;
  border: none;
  border-radius: 0;
  background: transparent;
}

.espnet3-guide .config-field {
  margin-bottom: 1.25rem;
}

.espnet3-guide .config-label {
  display: block;
  margin-bottom: 0.4rem;
  color: var(--text2);
  font-size: 12px;
}

.espnet3-guide .config-label span,
.espnet3-guide .config-section-label,
.espnet3-guide .sbatch-title {
  color: var(--text3);
}

.espnet3-guide .config-divider {
  height: 1px;
  margin: 1.25rem 0;
  background: var(--border);
}

.espnet3-guide .config-section-label,
.espnet3-guide .sbatch-title {
  margin-bottom: 0.75rem;
  letter-spacing: 0.08em;
  text-transform: uppercase;
}

.espnet3-guide select,
.espnet3-guide input[type="text"],
.espnet3-guide input[type="number"] {
  width: 100%;
  padding: 0.55rem 0.65rem;
  border: 1px solid var(--border2);
  border-radius: 6px;
  background: white;
  color: var(--text);
}

.espnet3-guide select:focus,
.espnet3-guide input:focus {
  border-color: var(--driver);
  outline: none;
}

.espnet3-guide .sbatch-preview {
  padding: 0.85rem 1rem;
  white-space: pre;
}

.espnet3-guide .sbatch-dir {
  color: var(--worker);
}

.espnet3-guide .sbatch-val {
  color: var(--provider);
}

.espnet3-guide .slurm-grid {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 0.75rem;
}

.espnet3-guide .ph-local .provider-half-header {
  background: var(--provider-bg);
  color: var(--provider);
}

.espnet3-guide .ph-worker .provider-half-header {
  background: var(--worker-bg);
  color: var(--worker);
}

.espnet3-guide .timing-badge {
  margin-left: auto;
  padding: 2px 7px;
  border-radius: 3px;
}

.espnet3-guide .ph-local .timing-badge {
  border: 1px solid var(--provider-border);
  background: var(--provider-bg);
  color: var(--provider);
}

.espnet3-guide .ph-worker .timing-badge {
  border: 1px solid var(--worker-border);
  background: var(--worker-bg);
  color: var(--worker);
}

.espnet3-guide .step-line {
  display: flex;
  align-items: flex-start;
  gap: 0.75rem;
  margin-bottom: 0.75rem;
  font-size: 13px;
}

.espnet3-guide .step-dot,
.espnet3-guide .step-ind {
  display: flex;
  align-items: center;
  justify-content: center;
  flex-shrink: 0;
}

.espnet3-guide .step-dot {
  width: 20px;
  height: 20px;
  margin-top: 2px;
  border-radius: 50%;
  font-size: 10px;
}

.espnet3-guide .dot-provider {
  border: 1px solid var(--provider-border);
  background: var(--provider-bg);
  color: var(--provider);
}

.espnet3-guide .dot-worker {
  border: 1px solid var(--worker-border);
  background: var(--worker-bg);
  color: var(--worker);
}

.espnet3-guide .stepper-header {
  margin-bottom: 1.5rem;
}

.espnet3-guide .step-ind {
  width: 28px;
  height: 28px;
  border: 1px solid var(--border2);
  border-radius: 50%;
  color: var(--text3);
}

.espnet3-guide .step-ind.active {
  border-color: var(--driver-border);
  background: var(--driver-bg);
  color: var(--driver);
}

.espnet3-guide .step-ind.done {
  border-color: var(--done-border);
  background: var(--done-bg);
  color: var(--done);
}

.espnet3-guide .role-card {
  position: relative;
  overflow: hidden;
  padding: 1.5rem;
}

.espnet3-guide .role-card::before {
  content: "";
  position: absolute;
  top: 0;
  right: 0;
  left: 0;
  height: 3px;
}

.espnet3-guide .role-card.c-driver::before {
  background: var(--driver);
}

.espnet3-guide .role-card.c-worker::before {
  background: var(--worker);
}

.espnet3-guide .role-card.c-provider::before {
  background: var(--provider);
}

.espnet3-guide .role-card.c-forward::before {
  background: var(--forward);
}

.espnet3-guide .role-label {
  margin-bottom: 0.5rem;
  font-size: 11px;
  letter-spacing: 0.08em;
  text-transform: uppercase;
}

.espnet3-guide .role-card h3 {
  margin-bottom: 0.5rem;
  font-size: 16px;
  font-weight: 500;
}

.espnet3-guide .role-card p {
  font-size: 13px;
  line-height: 1.65;
}

.espnet3-guide .role-methods {
  margin-top: 1rem;
  gap: 0.4rem;
}

.espnet3-guide .method-tag {
  padding: 2px 8px;
  border: 1px solid var(--border2);
  border-radius: 3px;
  background: var(--bg3);
  font-size: 11px;
}

.espnet3-guide .legend {
  gap: 0.6rem;
  margin-bottom: 2rem;
}

.espnet3-guide .legend-item {
  align-items: center;
  font-size: 12px;
}

.espnet3-guide .legend-dot {
  width: 10px;
  height: 10px;
  flex-shrink: 0;
  border-radius: 50%;
}

.espnet3-guide .ld-driver {
  background: var(--driver);
}

.espnet3-guide .ld-worker {
  background: var(--worker);
}

.espnet3-guide .ld-provider {
  background: var(--provider);
}

.espnet3-guide .ld-forward {
  background: var(--forward);
}

.espnet3-guide .ld-done {
  background: var(--done);
}

@media (max-width: 720px) {
  .espnet3-guide .container {
    padding: 0 1rem;
  }

  .espnet3-guide .exec-diagram,
  .espnet3-guide .config-grid,
  .espnet3-guide .overview-grid {
    grid-template-columns: 1fr;
  }

  .espnet3-guide .exec-diagram {
    flex-direction: column;
  }

  .espnet3-guide .exec-col {
    border-right: none;
    border-bottom: 1px solid var(--border);
  }

  .espnet3-guide .exec-col:last-child {
    border-bottom: none;
  }
}
</style>

<style scoped>
.espnet3-guide {
  min-height: 100vh;
}
</style>
