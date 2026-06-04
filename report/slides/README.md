# Conference Slides

Beamer slide deck summarizing the GPU GP-GOMEA report
(`../main.tex`). Uses the [CleanEasy](https://github.com/zemarchezi/CleanEasy_BeamerTheme)
theme, whose `.sty` files are vendored here so the deck builds without a
system-wide theme install.

## Build

```bash
tectonic slides.tex
```

(or `xelatex slides.tex` — XeTeX is required for the `xeCJK` author names).

## Contents

- `slides.tex` — deck source.
- `figs/` — figures extracted from the compiled report.
- `beamer*CleanEasy.sty` — vendored CleanEasy theme.
