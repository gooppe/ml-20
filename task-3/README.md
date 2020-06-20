# Task #3

ddpg работает плохо, td3 работает лучше.

Для запуска ddpg:

```bash
python ddpg.py --env swimmer swimmer6 --dump ddpg/model
```

Для запуска td3:

```bash
python td3.py --env swimmer swimmer6 --dump td3/model
```

Посмотреть визуализацию:

```bash
python visualize.py --env swimmer swimmer6 --dump td3/model_5000.pth
```
