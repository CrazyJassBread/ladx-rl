# Tips

## command
Here are some useful commands:

### tensorboard
To visualize training progress, you can use TensorBoard. Run the following command in your terminal:
```bash
tensorboard --logdir=logs
```
eg:
```bash
tensorboard \
  --logdir artifacts/tail_cave/room0a_three_of_a_kind \
  --host 127.0.0.1 \
  --port 6006
```