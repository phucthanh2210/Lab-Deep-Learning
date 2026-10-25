# Lịch sử triển khai nội bộ (không thuộc báo cáo nộp)

Các trạng thái chưa chạy GPU bên dưới là lịch sử trước khi tiếp nhận artifacts Kaggle. Kết quả hiện hành nằm trong walkthrough và validation/artifact_audit.json.

# Lab 03 — Báo cáo triển khai và trạng thái thực nghiệm

## 1. Đặc tả và thiết kế

### 1.0. Cách hiểu đề bài

Exercise 1 dùng mô hình sentiment pretrained trên Hugging Face, tokenize câu và dự đoán cảm xúc. Exercise 2 tải dữ liệu nhị phân, tải backbone/tokenizer, tiền xử lý, cấu hình TrainingArguments, fine-tune qua Trainer và đánh giá.

Theo yêu cầu mở rộng, bài dùng đầy đủ Rotten Tomatoes, ba notebook, run tracking độc lập, năm biểu đồ và phân tích lỗi. Không đặt ngưỡng accuracy giả định; mục tiêu đầu tiên là thực hiện và đo đúng quy trình.

### 1.1. Data/model contract

- Official splits: 8.530 train, 1.066 validation, 1.066 test. Loader kiểm tra kích thước và nhãn 0/1; không lấy subset.
- Tokenization: tối đa 128 tokens, truncation, dynamic padding. Batch có `input_ids`, `attention_mask`, `labels`.
- Backbone: DistilBERT; logits `(batch_size, 2)`, cross-entropy loss, AdamW.
- Pretrained sentiment model của Exercise 1 được phân biệt với backbone chưa fine-tune của Exercise 2.
- Chọn checkpoint bằng validation accuracy. Test chỉ đánh giá sau huấn luyện, không dùng để chọn siêu tham số.

### 1.2. Tái lập và lưu artifacts

Seed 42 qua Hugging Face `set_seed`, bao gồm Python/NumPy/PyTorch; cuDNN benchmark tắt, deterministic bật. Không khẳng định seed tạo ra kết quả giống hệt trên mọi thiết bị hoặc phiên bản.

Config trực tiếp được pin phiên bản thư viện; revision Hub có thể đặt SHA trong config. Run ghi môi trường, model commit và dataset fingerprints. Mỗi run có UUID, UTC timestamp, config, console log, trạng thái, SHA256 manifest. Registry lưu active run thành công gần nhất và lịch sử cả run lỗi.

## 2. Triển khai và kiểm chứng

### 2.1. Các thay đổi so với bản tham khảo

- IMDB subset được thay bằng toàn bộ Rotten Tomatoes và giữ nguyên official validation split.
- Notebook/CLI dùng chung workflow; không có hai cách chia dữ liệu hoặc hai quy trình huấn luyện khác nhau.
- Sanity overfit dùng mô hình disposable trong tests; mô hình train chính tải mới.
- CLI suy luận mặc định dùng mô hình sentiment pretrained; đánh giá yêu cầu checkpoint có thật, không âm thầm quay về backbone.
- FP16 chỉ bật trên CUDA. Warmup ratio trong config thực sự được truyền cho TrainingArguments.
- Có train/validation learning curves, confusion matrix số lượng và phần trăm chuẩn hóa, metrics bar, token-length KDE/cutoff, bảng dự đoán đúng/sai.
- Accuracy và binary/macro Precision/Recall/F1 tính tại máy bằng scikit-learn. `evaluate` được cài theo đề và ghi phiên bản, không tải code metric từ Hub.

### 2.2. Bằng chứng kiểm thử đã chạy

Môi trường kiểm thử: Linux, CPython 3.12.14, PyTorch 2.6.0+cpu, Transformers 4.48.3, Datasets 3.2.0. Không có GPU.

Lệnh chạy tại `Lab3/`:

```bash
python -m pytest -q
```

Kết quả: **9 passed**. Các test dùng dữ liệu fixture và DistilBERT rất nhỏ được khởi tạo cục bộ; không tải mô hình/dataset trên Hub.

1. Kiểm tra hợp đồng đủ split và từ chối split thiếu mẫu bằng dataset fixture.
2. Kiểm tra tokenization, truncation, dynamic padding và labels.
3. So sánh xác suất/nhãn pipeline với manual logits.
4. Kiểm tra lưu/tải checkpoint giữ nguyên dự đoán.
5. Sanity overfit một mini-batch: loss < 0.05 trên mô hình nhỏ dùng riêng.
6. Trainer train/evaluate, checkpoint tốt nhất và metrics.
7. Run tracking: không ghi đè, console log, manifest, active registry khi run lỗi.
8. Xuất đủ năm file PNG.
9. Workflow từ đầu đến cuối trên fixture: train, evaluate, checkpoint reload, JSON predictions, report và năm biểu đồ.

Kiểm thử bắt được và đã sửa lỗi trong callback đo train accuracy: `evaluate()` có thể reset cờ điều khiển của Trainer, làm mất lịch validation/save. Callback hiện bảo toàn cờ; test xác nhận `best_model_checkpoint` thực sự tồn tại.

`pip check` không phát hiện xung đột dependency. Các cảnh báo deprecation từ Matplotlib/Pyparsing và Datasets được giữ nguyên, không tắt assertions hoặc bỏ qua test lỗi.

### 2.3. Phần chưa thể kiểm chứng trong máy cloud

Truy vấn Hugging Face Hub bị proxy trả `403 Forbidden`; máy cũng không có CUDA. Vì vậy:

- Chưa tải và xác nhận trực tiếp dữ liệu thật trong máy này.
- Chưa chạy suy luận với trọng số sentiment thật.
- Chưa fine-tune toàn bộ 8.530 mẫu và chưa có Accuracy/F1 test thực của bản mới.
- Chưa có năm biểu đồ hoặc phân tích lỗi từ một run Rotten Tomatoes thật. Tests đã xác minh khả năng tạo chúng với fixture, không thay thế thực nghiệm.

Không dùng số liệu IMDB 86.4%/86.5% hoặc các số liệu cũ để gán cho bài này. Notebook không chèn output chưa thực thi.

## 3. Quy trình hoàn tất báo cáo thực nghiệm

Chạy trên Kaggle/Colab hoặc máy GPU có truy cập Hugging Face, theo README:

1. Cài requirements và xác nhận CUDA.
2. Chạy tests, rồi Exercise 1.
3. Chạy `python Lab3/scripts/train_finetune.py` từ gốc repo: đủ official splits, ba epochs, chọn checkpoint theo validation.
4. Mở run hoàn thành mà `config/workflow.json` trỏ tới. Kiểm tra `status.json`, metrics, provenance, reload check và SHA256 manifest.
5. Đọc `report.md` và `sample_predictions.json`; bổ sung phân tích riêng cho những câu thực sự bị đoán sai, phân biệt quan sát với giả thuyết.
6. Đồng bộ số liệu trong báo cáo nộp với đúng run ID; lưu checkpoint ngoài Git, chỉ chia sẻ artifacts nhẹ cần thiết.

Mã nguồn đáp ứng phạm vi triển khai; chất lượng mô hình trên dataset thật vẫn cần thực nghiệm GPU. Không xem kiểm thử offline là bằng chứng đã hoàn tất fine-tuning thật.

## Cập nhật kiểm thử trên Kaggle GPU

Test Trainer trước đây giả định global_step=4 và FP16 luôn tắt, chỉ đúng với môi trường CPU đã kiểm chứng. Đã sửa để tính số bước từ độ dài dataloader thực tế (bao gồm nhiều GPU) và kiểm tra FP16 theo yêu cầu cấu hình cùng CUDA availability. Thêm hai trường hợp yêu cầu FP16 bật/tắt. Sau sửa: **10 passed trên CPU**; cần chạy lại trên Kaggle để xác minh GPU, không khẳng định đã kiểm chứng GPU tại máy cloud.

## Sửa kiểm tra checkpoint sau fine-tuning GPU

Log Kaggle do người dùng cung cấp cho run `20261025T174902Z_finetuning_1e3ae3c9` ghi nhận đủ official splits và hoàn thành ba epochs (1.602 optimizer steps), test accuracy 0.8395872420 và binary F1 0.8406337372. Sau đó run lỗi ở assertion reload cũ, nên chưa được ghi nhận hoàn tất toàn bộ pipeline.

Kiểm tra cũ so sánh live model của Trainer/Accelerate trên CUDA với model tải lại trên CPU, có thể khác ngữ cảnh autocast và sai số số học. Bản sửa kiểm tra tensor trọng số khớp chính xác trước, rồi so sánh suy luận trên hai instance sạch trong cùng CPU FP32. Không nới tolerance để che lỗi trọng số.

Thêm script khôi phục riêng, chỉ dùng khi đủ artifacts/epochs và lỗi reload đã biết. Giữ log lỗi, kiểm tra manifest trước khi khôi phục, kiểm tra best checkpoint, tạo report và cập nhật registry/manifest mà không train lại. Kiểm thử CPU: **11 passed**, gồm forward wrapper, phát hiện trọng số sai và khôi phục run fixture. Việc khôi phục run Kaggle thật vẫn cần thực thi tại máy của người dùng.

## Rà soát nghiệm thu cuối

Chạy lại ba notebook view offline, xác nhận registry/artifacts không thay đổi; audit 24 file, tính lại metrics từ 1.066 dự đoán; kiểm tra năm PNG, CLI help và lỗi thiếu checkpoint. Phát hiện eval/infer dùng config hiện tại khi đánh giá checkpoint cũ: đã ưu tiên config snapshot của run, bỏ bước tải base tokenizer không cần thiết trong eval. Test CLI thực thi với model fixture và dataset mock xác nhận không tải backbone, dùng đúng dataset và max_length gốc. Bộ kiểm thử hiện tại: 14 passed trên CPU. Chi tiết kiểm tra: validation/final_review.json.
