import { defineClientConfig } from "vuepress/client";
import HomeCapabilities from "./components/HomeCapabilities.vue";
import HomeContribute from "./components/HomeContribute.vue";
import HomeDocLinks from "./components/HomeDocLinks.vue";
import HomeHero from "./components/HomeHero.vue";
import HomePipeline from "./components/HomePipeline.vue";
import HomeRecipe from "./components/HomeRecipe.vue";
import HomeQuickStart from "./components/HomeQuickStart.vue";
import DocCard from "./components/DocCard.vue";
import DocCards from "./components/DocCards.vue";
import FileStageMappingWidget from "./components/FileStageMappingWidget.vue";
import DataloaderDemo from "./components/DataloaderDemo.vue";
import ESPnet3ParallelGuide from "./components/ESPnet3ParallelGuide.vue";
import MetricsExplorer from "./components/MetricsExplorer.vue";
import DemoConfigEditor from "./components/DemoConfigEditor.vue";

export default defineClientConfig({
  enhance({ app }) {
    app.component("HomeCapabilities", HomeCapabilities);
    app.component("HomeContribute", HomeContribute);
    app.component("HomeDocLinks", HomeDocLinks);
    app.component("HomeHero", HomeHero);
    app.component("HomePipeline", HomePipeline);
    app.component("HomeRecipe", HomeRecipe);
    app.component("HomeQuickStart", HomeQuickStart);
    app.component("DocCard", DocCard);
    app.component("DocCards", DocCards);
    app.component("FileStageMappingWidget", FileStageMappingWidget);
    app.component("DataloaderDemo", DataloaderDemo);
    app.component("ESPnet3ParallelGuide", ESPnet3ParallelGuide);
    app.component("MetricsExplorer", MetricsExplorer);
    app.component("DemoConfigEditor", DemoConfigEditor);
  },
});
