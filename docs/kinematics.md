# Inverse Kinematics

- [`jax`](https://github.com/jax-ml/jax)
- [`pyroki`](https://github.com/chungmin99/pyroki)
- every stroke tatbot executes is a sequence of joint angles
- joint angles are computed using inverse kinematics (ik)
- ik is computed on the GPU in parallel

see:

- `src/tatbot/data/stroke.py`
- `src/tatbot/gen/batch.py`