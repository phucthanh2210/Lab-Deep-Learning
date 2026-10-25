# Lab 03 — Getting started with Hugging Face

Thực hiện cả hai bài trong đề giảng viên: phân tích cảm xúc bằng mô hình pretrained và fine-tune mô hình phân loại văn bản nhị phân bằng Hugging Face Trainer.

## Phạm vi và dữ liệu

| Thành phần | Thiết lập |
|---|---|
| Exercise 1 | `distilbert/distilbert-base-uncased-finetuned-sst-2-english` |
| Exercise 2 | `distilbert/distilbert-base-uncased`, đầu phân loại hai nhãn |
| Dataset | `cornell-movie-review-data/rotten_tomatoes` |
| Train / validation / test | **8.530 / 1.066 / 1.066**, toàn bộ splits chính thức |
| Tokenization | WordPiece, truncation 128, dynamic padding |
| Fine-tuning | 3 epochs, batch 16/32, AdamW, LR 2e-5, weight decay 0.01, warmup ratio 0.1 |
| Chọn checkpoint | Validation accuracy, đánh giá/lưu mỗi epoch |
| Precision | FP16 khi có CUDA; CPU dùng FP32 |

Không lấy mẫu nhỏ, không tự chia lại validation, không dùng test để chọn checkpoint. Sanity overfit chỉ nằm trong tests trên mô hình dùng riêng; mô hình fine-tuning được khởi tạo mới.

## Cài đặt

Khuyến nghị **Python 3.12** (đã kiểm chứng 3.12.14 trên Linux). Từ thư mục gốc repository:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
# Trên Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -r Lab3/requirements.txt
```

Với GPU CUDA 12.4, cài PyTorch tương ứng trước `requirements.txt`:

```bash
python -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
python -m pip install -r Lab3/requirements.txt
```

Máy chỉ có CPU có thể cài wheel nhẹ hơn bằng `--index-url https://download.pytorch.org/whl/cpu`. Luôn kiểm tra driver và `torch.cuda.is_available()` trước huấn luyện. Không cần torchvision cho bài NLP này.

Các phiên bản phụ thuộc trực tiếp được pin trong `requirements.txt`; đây chưa phải lockfile toàn bộ phụ thuộc bắc cầu. Mỗi run ghi phiên bản thực tế. Các trường revision mặc định là `main`; muốn cố định thí nghiệm, thay bằng commit Hugging Face đã xác minh. Run lưu model commit và dataset fingerprints để truy vết.

## Chạy từ thư mục gốc repo

```bash
# Tests offline: mô hình DistilBERT nhỏ, dữ liệu fixture; không tải Hub
cd Lab3
python -m pytest -q
cd ..

# Exercise 1: mô hình sentiment thực, tải từ Hub
python Lab3/scripts/run_exercise1.py

# Exercise 2: toàn bộ dữ liệu, mặc định yêu cầu GPU
python Lab3/scripts/train_finetune.py

# Suy luận mặc định dùng mô hình sentiment đã fine-tune trên SST-2
python Lab3/scripts/infer.py --text "This movie was wonderful!"

# Đánh giá checkpoint fine-tuned: thiếu checkpoint sẽ báo lỗi, không dùng backbone thay thế
python Lab3/scripts/eval.py --model-path Lab3/outputs/runs/<run_id>/best_model
python Lab3/scripts/infer.py --model-path Lab3/outputs/runs/<run_id>/best_model --text "A disappointing film."
```

Nếu chủ động chấp nhận thời gian huấn luyện CPU dài, CLI hỗ trợ `--allow-cpu`. Unit test chỉ huấn luyện mô hình rất nhỏ; kết quả test không phải chất lượng mô hình trên Rotten Tomatoes.

Mạng cần truy cập `huggingface.co` và các host CDN/Xet mà Hub chuyển hướng tới. Dataset và mô hình công khai không yêu cầu token. Nếu proxy chặn, cập nhật allowlist thay vì tắt TLS hoặc chèn token vào code.

## Ba notebook

1. [`01_sentiment_analysis_pipeline.ipynb`](notebooks/01_sentiment_analysis_pipeline.ipynb): Exercise 1, bảng tokens/IDs/mask và so sánh pipeline/manual.
2. [`02_finetuning_binary_classification.ipynb`](notebooks/02_finetuning_binary_classification.ipynb): Exercise 2, kiểm tra splits, preprocessing, Trainer, metrics và biểu đồ.
3. [`lab03_huggingface_complete.ipynb`](notebooks/lab03_huggingface_complete.ipynb): tích hợp cả hai bài để thực hành/nộp.

Mở notebook trong checkout và chọn kernel đã cài requirements. Có thể chạy từ root, `Lab3/` hoặc `Lab3/notebooks/`. Với Kaggle/Colab: clone repository vào máy GPU, bật Internet, cài requirements phù hợp và mở notebook trong checkout. Notebook cần đi cùng `src/` và `config/`; không upload riêng một file rồi kỳ vọng module có sẵn.

Các cell dùng chung `src/workflow.py` với CLI để tránh lệch cách chia dữ liệu và huấn luyện. Notebook chưa chèn output full-training vì chưa có một lần chạy đầy đủ được xác minh.

## Cấu trúc

```text
Lab3/
  config/       config.json, workflow.json
  notebooks/    ba notebook
  scripts/      run_exercise1.py, train_finetune.py, eval.py, infer.py
  src/
    datasets/   tải và kiểm tra đủ official splits, tokenization
    models/     khởi tạo mô hình, pipeline/manual inference
    training/   Trainer, metrics, sanity test helper
    utils/      run tracking, JSON, environment snapshot, plots
    workflow.py quy trình dùng chung
  tests/        dataset, Exercise 1, checkpoint, Trainer, runs, plots
  outputs/      sinh tại runtime, không commit checkpoint/cache
```

## Artifacts và registry

Mỗi lần chạy tạo `outputs/runs/<UTC_timestamp>_<type>_<uuid>/` độc lập. `config/workflow.json` ghi lịch sử và active run hoàn thành gần nhất; run lỗi không thay thế active run thành công. Registry dùng file lock và thay thế JSON nguyên tử. Commit registry cập nhật chỉ khi muốn chia sẻ metadata; checkpoint thực tế vẫn phải được lưu ngoài Git.

Run fine-tuning xuất:

- `config.json`, `environment_snapshot.json`, `provenance.json`, `console.log`, `status.json`, `manifest.json` (SHA256).
- `data_statistics.json`, `trainer_history.json`, `train_metrics.json`, `test_metrics.json`.
- **5 biểu đồ:** `learning_curves.png` (train/validation loss, train/validation accuracy), `confusion_matrix.png` (số lượng và % theo hàng), `metrics_bar.png`, `token_length_distribution.png` (KDE và cutoff), `sample_predictions.png` (bảng đúng/sai tô màu).
- `all_predictions.json`, `sample_predictions.json`, `report.md`: ví dụ đúng/sai thật từ run. Báo cáo tự động đưa ra giả thuyết theo dấu hiệu ngôn ngữ; cần đọc và bổ sung phân tích thủ công trước khi nộp.
- `best_model/`, `reload_check.json`: lưu và kiểm tra lại checkpoint.

Accuracy và cả binary/macro Precision, Recall, F1 được tính bằng scikit-learn tại máy; `evaluate` vẫn được cài và ghi phiên bản theo đề, nhưng không cần tải metric scripts từ Hub. Confidence không phải bằng chứng về nguyên nhân dự đoán. Learning curve training loss là loss log trong quá trình tối ưu; validation loss được tính khi model ở eval mode.

## Tình trạng kiểm chứng

Xem [báo cáo](walkthrough_lab_03.md). Không có số liệu Accuracy/F1 full-training được khẳng định trước khi chạy thành công trên dataset thật. Các số 86.4%/86.5% của IMDB và 84.15% từ tài liệu cũ không phải kết quả của repository này.

## Đối chiếu tiêu chí

| Tiêu chí yêu cầu | Triển khai |
|---|---|
| Toàn bộ Rotten Tomatoes | Loader kiểm tra đúng 8.530/1.066/1.066 |
| Ba notebook | Một cho mỗi bài và một tích hợp |
| Isolated runs + registry | UUID, UTC, snapshot, log, SHA256, active runs |
| Năm biểu đồ | Tạo từ dữ liệu thực của mỗi run |
| Phân tích lỗi | JSON toàn bộ dự đoán, mẫu đúng/sai, báo cáo theo run |
| Pytest | Tests offline bao gồm pipeline/manual, sanity và end-to-end fixture |
| JSON config | `config.json`, `workflow.json` |
| Sanity gate | Chỉ trong tests, không thay đổi trọng số trước train chính |

## Khôi phục run lỗi ở bước reload cũ

Nếu train/evaluate đã hoàn tất nhưng phiên bản cũ báo `Checkpoint reload predictions differ`, không cần train lại:

```bash
python Lab3/scripts/recover_run.py --run-dir Lab3/outputs/runs/<run_id>
```

Script kiểm tra SHA256 artifacts gốc, đủ epochs và 1.066 dự đoán test, so sánh chính xác trọng số `best_model` với checkpoint được Trainer chọn, rồi kiểm tra suy luận trong cùng CPU FP32. Chỉ khi kiểm tra đạt mới tạo báo cáo và cập nhật trạng thái/registry. Log lỗi gốc được giữ, thêm `recovery.log` và `recovery.json`, cập nhật manifest. Không đổi metrics và không huấn luyện lại.
