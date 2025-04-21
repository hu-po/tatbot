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

test ik functionality

```bash
./scripts/test/ik.sh
./scripts/test/ik.ai.sh # this will test model apis
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