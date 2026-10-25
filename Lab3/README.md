# Lab 03 — Getting started with Hugging Face

Bài thực hành Deep Learning — UTH: sentiment analysis bằng mô hình pretrained và fine-tuning mô hình phân loại văn bản nhị phân bằng Hugging Face Trainer.

**Đã có kết quả Kaggle thật:** dùng toàn bộ Rotten Tomatoes, 8.530 train / 1.066 validation / 1.066 test. Test Accuracy **83,96%**, Binary F1 **84,06%**. Mã nguồn, notebook và báo cáo dùng chung run registry, có thể xem kết quả mà không train lại.

## 1. Xem bài và kết quả

- [Notebook Exercise 1](notebooks/01_sentiment_analysis_pipeline.ipynb): tokenization, pipeline và manual forward.
- [Notebook Exercise 2](notebooks/02_finetuning_binary_classification.ipynb): dữ liệu, Trainer, metrics, năm biểu đồ và phân tích lỗi.
- [Notebook tích hợp](notebooks/lab03_huggingface_complete.ipynb): cả hai Exercise, phù hợp xem toàn bộ bài.
- [Báo cáo thực hành](walkthrough_lab_03.md): lý thuyết, phương pháp, kết quả và giới hạn có bằng chứng.
- [Đề bài](description_lab_03.md).

Các notebook đã lưu output của một lượt thực thi **`MODE="view"`** trên artifacts Kaggle. Có thể đọc trực tiếp trên GitHub. Khi chạy lại notebook, mặc định view đọc dữ liệu đã commit, không tải model/dataset, không huấn luyện và không cần GPU. Đổi `MODE="train"` ở cell setup để tạo thí nghiệm mới; phần fine-tuning yêu cầu GPU theo mặc định.

## 2. Cấu trúc

```text
Lab3/
  config/
    config.json             tham số cho lần chạy mới
    workflow.json           active runs và lịch sử
  notebooks/                ba notebook có output
  src/
    datasets/               tải official splits, kiểm tra kích thước và tokenization
    models/                 model, pipeline/manual, kiểm tra checkpoint
    training/               Trainer, metrics, sanity overfit dùng riêng trong tests
    utils/                  log, registry, manifest và plots
    workflow.py             luồng CLI/notebook dùng chung
    evidence.py             audit đọc-only, tính lại metrics từ dự đoán
  scripts/                  train, eval, infer, audit và recovery
  tests/                    tests chức năng và kiểm tra artifacts thật
  outputs/runs/             artifacts Kaggle nhẹ, giữ nguyên manifest gốc
  validation/               kết quả audit và thực thi notebook tại máy rà soát
  description_lab_03.md     đề bài
  walkthrough_lab_03.md     báo cáo thực nghiệm để nộp
  internal_changes.md       lịch sử kỹ thuật; không thuộc báo cáo nộp
```

Đã hợp nhất các thư mục `config/` và `outputs/` được upload ở gốc repo vào `Lab3/`. Không có hai registry độc lập. Các byte artifacts gốc không bị chỉnh sửa; 24 file hiện có khớp manifest.

## 3. Cài đặt

Khuyến nghị Python **3.12**: đã kiểm chứng CPU Linux 3.12.14 và run Kaggle ghi nhận 3.12.15. Dùng môi trường riêng, tránh hạ cấp thư viện nền Kaggle.

Từ gốc repo, trên Linux/Kaggle:

```bash
python -m pip install uv
uv python install 3.12
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r Lab3/requirements.txt
uv pip check --python .venv/bin/python
```

Trên Windows dùng `.venv\Scripts\python.exe` thay cho `.venv/bin/python`. Không cần kích hoạt nếu gọi trực tiếp interpreter.

Trên Kaggle, bật **Settings -> Accelerator -> GPU** và **Internet** trước training; session restart sẽ mất biến Python và cần chạy lại các cell setup. Có thể đặt venv ngoài repo tại `/kaggle/working/lab03-py312` như phiên thực nghiệm đã chạy.

`torch==2.6.0` bản Linux PyPI cung cấp CUDA build phù hợp của phiên đã kiểm chứng. Nếu cần cài wheel cụ thể trước requirements:

```bash
# CUDA 12.4: cần driver hỗ trợ; kiểm tra bằng nvidia-smi
uv pip install --python .venv/bin/python torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
# Hoặc máy chỉ CPU dùng https://download.pytorch.org/whl/cpu
```

Requirements pin dependencies trực tiếp, chưa là lockfile toàn bộ dependency bắc cầu. Không cần torchvision. Dataset và model công khai không cần token; mạng phải truy cập được Hugging Face và các CDN/Xet chuyển hướng. Không tắt kiểm tra TLS khi gặp lỗi tải.

## 4. Kiểm thử và audit (không cần GPU)

```bash
cd Lab3
../.venv/bin/python -m pytest -q
cd ..
.venv/bin/python Lab3/scripts/audit_outputs.py
```

- **13 tests đạt trên CPU**: dataset contract, dynamic padding, pipeline/manual, checkpoint, sanity overfit, Trainer, run tracking, năm plots, recovery, metrics/artifact audit và phát hiện dữ liệu bị sửa.
- Audit kiểm tra 24 file SHA256, tính lại toàn bộ bảy metrics từ 1.066 predictions và xác nhận confusion matrix.
- 29 file checkpoint không có trong Git được liệt kê rõ là thiếu có chủ đích, không coi là đã kiểm tra tại máy rà soát.
- Ba notebook chạy hết ở view, Hugging Face offline, không có cell lỗi. Xem `validation/notebook_execution.json`.

Tests có fixture nhỏ để kiểm tra pipeline; chúng không phải benchmark Rotten Tomatoes. Audit thực dùng `all_predictions.json` từ Kaggle và không suy luận lại từ weights.

## 5. Chạy thí nghiệm mới trên GPU

Từ gốc repo, dùng interpreter của môi trường vừa cài:

```bash
.venv/bin/python Lab3/scripts/run_exercise1.py
CUDA_VISIBLE_DEVICES=0 TOKENIZERS_PARALLELISM=false .venv/bin/python Lab3/scripts/train_finetune.py
```

Trên Kaggle có thể gọi các lệnh bằng `subprocess.run([...], cwd=REPO, env=RUN_ENV, check=True)`, trong đó `RUN_ENV['CUDA_VISIBLE_DEVICES']='0'`; không dùng Python mặc định của notebook nếu requirements nằm trong venv khác.

Quy trình dùng toàn bộ official splits, max length 128, 3 epochs, batch train/eval 16/32, LR 2e-5, weight decay 0.01, warmup ratio 0.1, FP16 chỉ khi CUDA có sẵn. Lưu/evaluate mỗi epoch; chọn best theo validation accuracy. Sanity test không tác động mô hình train chính. Chạy lại tạo run mới, không resume hoặc ghi đè run cũ. Chỉ dùng `--allow-cpu` nếu chủ động chấp nhận thời gian full training CPU.

Có thể thực thi ba notebook với MODE=train thay CLI. Không chạy notebook tích hợp và notebook từng bài đồng thời nếu chỉ muốn một lần thí nghiệm.

## 6. Kết quả hiện hành và truy vết

| Giai đoạn | Run được registry chọn |
|---|---|
| Exercise 1 | `20261025T181532Z_exercise1_71da6f40` |
| Fine-tuning | `20261025T181550Z_finetuning_e02bced7` |

| Test metric | Giá trị |
|---|---:|
| Accuracy | 83,9587% |
| Binary Precision / Recall / F1 | 83,5185% / 84,6154% / 84,0634% |
| Macro Precision / Recall / F1 | 83,9646% / 83,9587% / 83,9580% |
| Đúng / tổng | 895 / 1.066 |

Confusion matrix `[TN FP; FN TP]` = `[[444,89],[82,451]]`. Checkpoint epoch 3 được chọn theo validation accuracy 84,33%; epoch 2 có loss validation thấp hơn. Có dấu hiệu overfitting; đề không đặt ngưỡng accuracy 85%.

Nguồn: [metrics](outputs/runs/20261025T181550Z_finetuning_e02bced7/test_metrics.json), [toàn bộ dự đoán](outputs/runs/20261025T181550Z_finetuning_e02bced7/all_predictions.json), [lịch sử](outputs/runs/20261025T181550Z_finetuning_e02bced7/trainer_history.json), [audit độc lập](validation/artifact_audit.json).

## 7. Checkpoint, inference và lưu trữ

**GitHub không chứa trọng số.** Run Kaggle lưu bằng chứng reload thành công trong `reload_check.json`; máy rà soát không chạy lại checkpoint. Không có URL weights công khai đã xác minh, nên cần giữ ZIP Kaggle của bạn.

Để đánh giá/suy luận lại, giải nén `best_model/` từ ZIP vào đúng đường dẫn:

```text
Lab3/outputs/runs/20261025T181550Z_finetuning_e02bced7/best_model/
```

Rồi chạy:

```bash
.venv/bin/python Lab3/scripts/eval.py --model-path Lab3/outputs/runs/20261025T181550Z_finetuning_e02bced7/best_model
.venv/bin/python Lab3/scripts/infer.py --model-path Lab3/outputs/runs/20261025T181550Z_finetuning_e02bced7/best_model --text "A wonderful movie."
```

Không truyền `--model-path` thì infer dùng sentiment pretrained SST-2 của Exercise 1, **không phải** mô hình Rotten Tomatoes vừa train. Eval không âm thầm fallback khi thiếu checkpoint.

SHA256 weights được giữ trong manifest: `5a63c9c2a6391132c86b0e6c08277cc23ccc692dd080b78ba2db074225bb5105`. Khi đưa weights vào lại, audit sẽ kiểm tra hash thật. Không upload weights/optimizer hoặc ZIP lớn vào Git.

## 8. Artifacts và chia sẻ run mới

Mỗi run có config, snapshot, console log, provenance, status, SHA256 manifest, metrics, lịch sử, dự đoán và report. Năm hình: learning curves, confusion matrix chuẩn hóa kèm counts, binary metrics bar, train token-length KDE/cutoff và bảng mẫu đúng/sai.

`outputs/` mặc định bị ignore để tránh đẩy nhầm model; các artifacts nhẹ của hai run hiện hành đã được commit. Với run mới, chỉ stage phần nhẹ:

```bash
git add -f -- Lab3/outputs/runs/<run_id> ':(glob,exclude)**/best_model/**' ':(glob,exclude)**/checkpoints/**'
git add Lab3/config/workflow.json
git diff --cached --stat
```

Giữ manifest gốc kể cả khi weights không đi cùng. Không chỉnh sửa artifacts lịch sử để làm số liệu đẹp hơn; kết quả rà soát bổ sung ghi ở `validation/`. Run registry cần đi cùng run directory khi chuyển máy.
