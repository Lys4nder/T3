```bash
source venv/bin/activate
```

```bash
export DYLD_LIBRARY_PATH=/opt/homebrew/opt/libomp/lib:$DYLD_LIBRARY_PATH
```

```bash
python main.py
```

```bash
DYLD_LIBRARY_PATH=/opt/homebrew/opt/libomp/lib:$DYLD_LIBRARY_PATH python compare.py
```

```bash
python -m src.compare_xgboost
```

```bash
DYLD_LIBRARY_PATH=/opt/homebrew/opt/libomp/lib:$DYLD_LIBRARY_PATH T3_RUN_XGB_COMPILED=1 python -m src.compare_xgboost
```

```bash
python -m src.compare_catboost
```

```bash
T3_RUN_CAT_COMPILED=1 python -m src.compare_catboost
```
