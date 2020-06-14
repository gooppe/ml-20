# DQN + LunarLander

Обучение:
```bash
python main.py

> States: 8; Actions: 4
> Episode 100 average score: -118.5721183796475
> Episode 200 average score: -65.89658549844043
> Episode 300 average score: -21.05148916317569
> Episode 400 average score: 60.19892396859911
> Episode 500 average score: 134.4213216211325
> Episode 600 average score: 196.53098497960255
> Task solved with 700 episodes. Average score: 229.47376548760846
> Episode 800 average score: 232.6512570033434
> Episode 900 average score: 251.04644482596532
> Episode 1000 average score: 252.95532374903985
```

![Scores](plot.png)

Тест и примеры полетов:
```bash
python test.py --dump 700model.pth

> Average score: 231.04605811898836
> 85 episodes with score > 200
> 96 episodes with score > 100
> 99 episodes with score > 0
> 1 episodes with score < 0
```

```bash
python test.py --dump 1000model.pth

> Average score: 253.2751754764696
> 97 episodes with score > 200
> 99 episodes with score > 100
> 100 episodes with score > 0
> 0 episodes with score < 0
```

![Examples](lander.gif)
