# Docstring Guide

Good docstrings help users, reviewers, and coding agents.

In ESPnet3, docstrings are part of the public developer experience.
Please treat them as required documentation, not optional polish.

::: important
Write docstrings for both `espnet3/` code and recipe code under `egs3/`.
Coding agents can write strong first drafts, so we expect substantial APIs to
be documented well.
:::

## What to optimize for

A good docstring should answer these questions quickly:

- What does this function or class do?
- When should someone use it?
- What inputs does it expect?
- What does it return or change?
- What can fail?
- What is a realistic example of calling it?

Prefer short English sentences.
Most developers will skim first and read in detail only when needed.

## Public APIs vs private helpers

Treat these as different documentation targets.

### Public or externally used APIs

Public functions, classes, methods, and externally consumed helpers should:

- explain both behavior and intended usage
- document important config expectations
- document special syntax and unusual invocation patterns
- include concrete examples when examples make usage clearer

For substantial public APIs, include these sections when they materially help:

- `Args`
- `Returns`
- `Raises`
- `Notes`
- `Examples`

::: note
For ESPnet3, examples are especially important. Put multiple examples if possible.
:::

### Private helpers

Private helpers, especially `_`-prefixed functions and methods, should stay
short.

Usually a one-line or short multi-line docstring is enough:

- what the helper does
- why a non-obvious branch or normalization exists

Do not add full public-API sections unless the helper is effectively used as a
shared external interface.

::: note
Too little documentation is a recurring problem.
Too much documentation is usually easier to trim than missing documentation is
to reconstruct later.
:::

## Style expectations

Follow the [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html).

For ESPnet3 docstrings, this usually means:

- start with a short summary line
- use clear section headers
- keep argument descriptions precise
- describe behavior, not implementation trivia
- show realistic examples, not toy placeholders

## What to document explicitly in ESPnet3

ESPnet3 has several patterns that deserve explicit docstring coverage.

Document these when relevant:

- stage names such as `train`, `infer`, `measure`, or recipe-specific stages
- config file expectations such as `training.yaml` or `inference.yaml`
- Hydra/OmegaConf-style fields such as `_target_` and `${...}`
- recipe-local import paths under `egs3`
- dataset field names like `speech`, `text`, `uttid`
- output directories such as `exp_dir` or `inference_dir`
- differences between shared `espnet3/` code and recipe-local `egs3/` code

If the API accepts a special form, show that form in an example.

## Good patterns

### Public function

```python
def load_dataset_module(module_name: str, recipe_dir: Path) -> type:
    """Load a dataset module from a recipe-local import path.

    This helper resolves recipe-local dataset modules used by ESPnet3 recipes.
    Use it when a config or runner needs to import a dataset implementation
    from `egs3/...` without hardcoding repository-relative paths.

    Args:
        module_name: Absolute Python module path such as
            `egs3.mini_an4.asr.dataset.builder`.
        recipe_dir: Root directory of the active recipe.

    Returns:
        The imported module object.

    Raises:
        ModuleNotFoundError: If the module path cannot be imported.
        ValueError: If `module_name` is not an absolute import path.

    Notes:
        Relative imports under `egs3/` should be avoided. Prefer explicit
        absolute import paths.

    Examples:
        >>> load_dataset_module(
        ...     "egs3.mini_an4.asr.dataset.builder",
        ...     Path("egs3/mini_an4/asr"),
        ... )
    """
```

### Private helper

```python
def _normalize_stage_name(name: str) -> str:
    """Normalize aliases so stage comparisons use one canonical form."""
```

## Weak patterns to avoid

Avoid docstrings that only repeat the function name:

```python
def train_model(cfg):
    """Train model."""
```

Avoid long implementation narratives that never explain how to call the API.

Avoid examples that hide the real ESPnet3 usage pattern.

For example, this is too vague:

```python
def run_stage(name):
    """Run a stage.

    Args:
        name: Stage name.
    """
```

It does not explain valid names, config requirements, or how the function is
used in the recipe flow.

## Recipe code also needs docstrings

Do not skip docstrings in recipe code just because it is under `egs3`.
Recipe-local builders, systems, custom stages, and data preparation helpers are
often the first examples that users, reviewers, and coding agents read.

## Suggested coding-agent prompt snippet

If you use coding agents for ESPnet3 development, this instruction snippet is a
good baseline:

```text
When writing docstrings for this repository, distinguish between public/external
APIs and private helpers.

For public or externally used functions/classes, explain not only what they do
but also how they are intended to be used. Include `Args`, `Returns`,
`Raises`, `Notes`, and `Examples` when that detail materially helps users.
Include 1-2 concrete examples for substantial APIs. If there is any special
syntax, resolver, stage name, config shape, or unusual invocation pattern,
include an example that shows that form explicitly.

For private helpers, especially `_`-prefixed functions and methods, keep
docstrings concise. Focus on what the helper does and why any non-obvious
branch or normalization exists. Do not add full public-API style sections
unless the helper is effectively externally consumed.

Follow the Google Python Style Guide. Prefer short English sentences.
```

## Related pages

<DocCards :cols="3">
  <DocCard
    title="CI and PR"
    desc="See review expectations before you open a pull request."
    icon="tabler:git-pull-request"
    href="./ci-and-pr.html"
  />
  <DocCard
    title="Adding a stage"
    desc="See how new stage methods are added and documented."
    icon="tabler:route"
    href="./adding-a-stage.html"
  />
  <DocCard
    title="Naming conventions"
    desc="Keep doc examples aligned with repository naming rules."
    icon="tabler:abc"
    href="./naming-conventions.html"
  />
</DocCards>
