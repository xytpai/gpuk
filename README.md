### Performance

| OP | DEVICE | SHAPE | TIME | PERF | PEAK | RATIO |
|--|--|--|--|--|--|--|
| hgemm-wmma | RTX4090 | m=8192,n=8192,k=8192 | 6.7 ms | 163.5 TFLOPS | 165.2 TFLOPS | 99% |
| hgemm-wmma | H20 | m=8192,n=8192,k=8192 | 12.2 ms | 89.5 TFLOPS | 95 TFLOPS | 94% |

### Install

```bash
git clone https://github.com/xytpai/gpuk
cd gpuk
python -m pip install -e . --no-build-isolation
```

### Test

```bash
python test/test_all_reduce_fusion.py
```
