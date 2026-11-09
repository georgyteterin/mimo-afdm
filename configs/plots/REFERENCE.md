# Plot config reference

Run: `python -m plotters.plot_ber --plot-cfg configs/plots/<name>.yaml`

The plotter checks every value before drawing and lists all errors at once.

## Top-level keys

| key | required | values | default |
|---|---|---|---|
| `curves` | yes | list of curves, see below | — |
| `ci` | no | `true` / `false`: 95% Clopper-Pearson intervals | `false` |
| `theory` | no | list from: `qpsk_awgn`, `qpsk_rayleigh` | `[]` |
| `title` | no | string or `null` | `null` |

## Curve keys

| key | required | values | default |
|---|---|---|---|
| `path` | yes | result JSON, `.json` may be omitted: `results/<sim cfg>/<test>` | — |
| `label` | no | any string | `test_name` from JSON |
| `color` | no | see Colors | `tab:blue` |
| `marker` | no | see Markers | cycles `o s ^ v D x * p < >` |
| `ls` | no | see Line styles | `-` |
| `lw` | no | positive number | `1.5` |

**Always quote `marker` and `ls`**: in YAML `*` starts an alias and `-` starts a list item.

```yaml
- path: results/fading_test/afdm_mmse
  label: AFDM MMSE
  marker: "*"
  color: tab:red
  ls: "--"
```

## Colors

Tableau palette (matplotlib default, readable in print):
`tab:blue`, `tab:orange`, `tab:green`, `tab:red`, `tab:purple`,
`tab:brown`, `tab:pink`, `tab:gray`, `tab:olive`, `tab:cyan`

There is no `tab:yellow`; closest is `tab:olive`.

Also valid: `black`, `C0`…`C9`, hex `"#1f77b4"`, any CSS name (`navy`, `darkred`, `gold`, ...):
https://matplotlib.org/stable/gallery/color/named_colors.html

## Markers

| value | shape | value | shape |
|---|---|---|---|
| `"o"` | circle | `"s"` | square |
| `"^"` | triangle up | `"v"` | triangle down |
| `"<"` | triangle left | `">"` | triangle right |
| `"D"` | diamond | `"d"` | thin diamond |
| `"p"` | pentagon | `"h"`, `"H"` | hexagon |
| `"8"` | octagon | `"*"` | star |
| `"P"` | filled plus | `"X"` | filled x |
| `"+"` | plus | `"x"` | x |
| `"1"`…`"4"` | tri down/up/left/right | `"."` | point |
| `","` | pixel | `"\|"`, `"_"` | vertical / horizontal line |

## Line styles

| value | alias | look |
|---|---|---|
| `"-"` | `solid` | ───── |
| `"--"` | `dashed` | ─ ─ ─ |
| `"-."` | `dashdot` | ─ · ─ · |
| `":"` | `dotted` | · · · · |

## Theory curves

| value | formula | source |
|---|---|---|
| `qpsk_awgn` | ½·erfc(√(Eb/N0)) | Proakis, *Digital Communications* |
| `qpsk_rayleigh` | ½·(1 − √(γ/(1+γ))), γ = Eb/N0 | Proakis, *Digital Communications* (flat Rayleigh) |
