<div align="center">
  <a href="https://tatbot.ai/">
    <picture>
      <source media="(prefers-color-scheme: dark)" srcset="assets/logos/dark.svg">
      <img src="assets/logos/light.svg" alt="tatbot">
    </picture>
  </a>
</div>

# **tatbot**: tattoo robot

<div align="center">
  <a href="https://github.com/hu-po/tatbot/blob/main/LICENSE"><img src="https://img.shields.io/github/license/hu-po/tatbot.svg?v" alt="license"></a>
</div>

tatbot is a robotic tattoo device.

- [`docs/story.md`](docs/story.md)
- [`docs/software.md`](docs/software.md)
- [`docs/hardware.md`](docs/hardware.md)
- [`docs/todo.md`](docs/todo.md)

## setup

configure and set the backend (e.g. `x86-3090`, `arm-rpi`)

```bash
source scripts/backends/x86-3090.sh
```

## workflow

the following assets are required:

- at least one tattoo design in `assets/designs`
- a 3d mesh model of the tattoo area in `assets/3d`
- a urdf of the robot arm in `assets/trossen-arm-description`

run the stencil simulation to generate ik poses

```bash
./scripts/stencil.sh
```

visualize the final stencil placement (requires `usdview`)

```bash
usdview $TATBOT_ROOT/output/stencil.usd
```

run the ik solver with a specific morph (e.g. `gpt-e409cb`)

```bash
./scripts/ik/morph.render.sh gpt-e409cb
```

visualize the ik result (requires `usdview`)

```bash
usdview $TATBOT_ROOT/output/ik_gpt-e409cb.usd
```

## testing

the following tests should work on all backends

```bash
./scripts/test/ik.sh
./scripts/test/ik.ai.sh # this will test model apis (do not run this every time as it consumes credits)
```

## Citation

```
@misc{tatbot-2025,
  title={tatbot},
  author={Hugo Ponte},
  year={2025},
  url={https://github.com/hu-po/tatbot}
}
```