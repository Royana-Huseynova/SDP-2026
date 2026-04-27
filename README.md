# How to Run the Program

The program is controlled via the terminal. You use the `cli.main` module to execute commands.

---

## 1. The Basic Command

To run a single experiment, use this syntax:

```bash
python -m cli.main --dataset [DATASET_NAME] --model [MODEL_NAME] --data_path [YOUR_PATH]
```

**Parameters:**

- `--dataset` — The name of the dataset you want to load.
- `--model` — The specific model you want to run.
- `--data_path` — The file path to your folder containing the images.

---

## 2. Running Multiple Models (The Batch Loop)

If you want to run several models in a row without typing the command every time, use this loop:

```bash
DATA_PATH="/path/to/your/data"
for m in model1 model2 model3; do
    python -m cli.main --dataset [DATASET_NAME] --model $m --data_path $DATA_PATH
done
```

---

## 3. Adding Metrics

If you want to calculate specific performance metrics, add the `--metrics` flag at the end:

```bash
python -m cli.main --dataset [NAME] --model [MODEL] --data_path [PATH] --metrics psnr ssim cpsnr sam ergas
```
