# Qwen Kaggle loader

Use this single Kaggle cell:

```python
import urllib.request

url = "https://raw.githubusercontent.com/annapozdn/apozdn/main/qwen_kaggle_app.py"
exec(urllib.request.urlopen(url).read().decode("utf-8"))
```

If the cell says dependencies were installed or changed:

1. Kaggle: `Draft Session` -> `More settings` -> `Restart & Clear Cell Outputs`
2. Run the same tiny loader cell again.

Do not run the old notebook cells or `Run All`.
