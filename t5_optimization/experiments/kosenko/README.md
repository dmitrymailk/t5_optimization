### 1 Experiment
Запуск дефолтной модели ai-forever/T5-large-spell на английском датасете.
```python
import torch
from sage.spelling_correction import (
    T5ModelForSpellingCorruption,
    RuM2M100ModelForSpellingCorrection,
    AvailableCorrectors,
)
from datetime import datetime

corrector = T5ModelForSpellingCorruption.from_pretrained(
    AvailableCorrectors.ent5_large.value
)

corrector.model.to(torch.device("cuda:3"))

start = datetime.now()
metrics = corrector.evaluate(
    "t5_optimization/sage/data/example_data/jfleg",
    batch_size=1,
    prefix="grammar: ",
)

duration = datetime.now() - start
print(duration)

print(metrics)
```

```text
0:17:08.273551
{'Precision': 83.39, 'Recall': 84.25, 'F1': 83.82}
```

Батч выбран равным 1 потому что при запуске пользователем у него будет только 1 запрос на исправление предложения.