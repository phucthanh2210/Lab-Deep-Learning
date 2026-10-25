# Báo cáo thực hành Lab 03 — Getting started with Hugging Face

**Môn học:** Deep Learning — UTH.
**Phạm vi:** Exercise 1 (sentiment inference) và Exercise 2 (binary text classification fine-tuning).
**Dữ liệu và mô hình:** toàn bộ Rotten Tomatoes, DistilBERT, Hugging Face Transformers/Datasets/Trainer.

## 1. Đặc tả, cơ sở lý thuyết và phương pháp

### 1.0. Diễn giải đề bài và sản phẩm

Exercise 1 yêu cầu cài Transformers, tải mô hình sentiment pretrained, tokenize câu và dự đoán cảm xúc. Exercise 2 yêu cầu cài Transformers/Datasets/Evaluate, tải dataset nhị phân, tải backbone/tokenizer, tiền xử lý, thiết lập TrainingArguments, tạo Trainer, fine-tune và đánh giá.

Bài làm bổ sung ba notebook, module dùng chung với CLI, JSON config, run tracking độc lập, kiểm thử và năm biểu đồ theo yêu cầu người dùng. Đề giảng viên không đặt ngưỡng Accuracy 85%; không lấy ngưỡng đó làm điều kiện sửa hoặc chọn kết quả test.

### 1.1. Bài toán và dữ liệu

Với văn bản đầu vào $x$, mô hình dự đoán $y \in \{0,1\}$, lần lượt Negative và Positive. Dataset `cornell-movie-review-data/rotten_tomatoes` được dùng đầy đủ, giữ nguyên các split chính thức:

| Split | Số mẫu | Vai trò |
|---|---:|---|
| Train | 8.530 | Cập nhật trọng số |
| Validation | 1.066 | Chọn checkpoint |
| Test | 1.066 | Đánh giá cuối sau lựa chọn |

Tập test có 533 mẫu mỗi lớp, xác nhận từ toàn bộ nhãn trong `all_predictions.json`. Không lấy subset hoặc tự chia lại validation. Việc giữ official splits tránh chủ động dùng test trong lựa chọn mô hình; chưa có kiểm toán duplicate/leakage toàn bộ văn bản train-validation-test trong artifacts hiện tại, nên không khẳng định đã chứng minh hoàn toàn không có trùng lặp ngữ nghĩa.

Tham chiếu đơn giản: luôn đoán Negative trên test cân bằng đạt Accuracy 50%, Macro F1 33,33%. Đây là baseline hằng từ phân bố nhãn, không phải kết quả của backbone DistilBERT chưa fine-tune. Không gán một accuracy giả định cho đầu phân loại khởi tạo ngẫu nhiên.

### 1.2. Transformer, DistilBERT và đầu phân loại

Self-attention kết hợp thông tin của các token:

$$\operatorname{Attention}(Q,K,V)=\operatorname{softmax}(QK^T/\sqrt{d_k})V.$$

DistilBERT là transformer encoder rút gọn từ BERT bằng knowledge distillation, có sáu lớp encoder trong backbone sử dụng. `AutoModelForSequenceClassification` bổ sung pre-classifier, activation, dropout và classifier hai lớp. Khi tải backbone vào bài phân loại, cảnh báo các trọng số classifier mới khởi tạo là bình thường; các trọng số đó cần được huấn luyện.

Với logits $z \in \mathbb{R}^2$, xác suất $p_c=e^{z_c}/\sum_j e^{z_j}$ và dự đoán $\hat y=\arg\max_c p_c$. Hàm mục tiêu là cross-entropy; tối ưu bằng AdamW. Logits có shape `(B,2)`; `input_ids`, `attention_mask` có shape `(B,S)`.

### 1.3. Tokenization và tiền xử lý

WordPiece ánh xạ văn bản sang vocabulary IDs, thêm `[CLS]` đầu chuỗi và `[SEP]` cuối chuỗi. Attention mask 1 đánh dấu token thật, 0 là padding. Truncation tối đa 128 tokens; dynamic padding bằng `DataCollatorWithPadding` đệm đến chiều dài lớn nhất trong từng batch.

Thống kê thực từ train: **trung bình 27,3683 tokens, tối đa 78, tỉ lệ bị cắt 0%**. Các số này có trong `data_statistics.json`; không suy rộng thống kê độ dài train sang hai split còn lại khi chưa đo.

### 1.4. Thiết kế thí nghiệm và tái lập

| Tham số | Giá trị |
|---|---|
| Backbone | `distilbert/distilbert-base-uncased` |
| Seed | 42 |
| Epochs | 3 |
| Batch train/eval | 16 / 32 |
| Learning rate | 2e-5 |
| Weight decay | 0,01 |
| Warmup ratio | 0,1 |
| Eval/save strategy | Mỗi epoch |
| Chọn best checkpoint | Validation accuracy |
| Mixed precision | FP16 trên CUDA |

Mô hình fine-tuning được tải mới; sanity overfit trong tests dùng mô hình disposable, không làm thay đổi trọng số train chính. Cố định seed bằng `set_seed`; cuDNN benchmark tắt, deterministic bật. Seed không đảm bảo mọi thiết bị/phiên bản cho kết quả bit-identical.

Môi trường Kaggle ghi nhận: **CPython 3.12.15, Tesla T4, PyTorch 2.6.0, Transformers 4.48.3, Datasets 3.2.0, Evaluate 0.4.3, Accelerate 1.3.0**. Metric thực tính bằng scikit-learn 1.6.1 tại máy, tránh tải metric scripts từ Hub. `evaluate` được cài và ghi phiên bản theo đề, nhưng không được dùng để tính các metric báo cáo.

Backbone commit thực: `12040accade4e8a0f71eabdb258fecc2e7e948be`. Dataset fingerprints train/validation/test lần lượt `0346a9d2308d9056`, `4d310508c191379e`, `23aa2ff3ad95b6f5`. Config Hub dùng revision `main`; fingerprints không thay thế dataset commit SHA. Vì chưa lưu dataset commit cụ thể, không khẳng định khả năng phục dựng tuyệt đối từ Hub nếu `main` thay đổi sau này.

## 2. Thực nghiệm và bằng chứng

### 2.1. Exercise 1 — Pretrained sentiment analysis

Run: [`20261025T181532Z_exercise1_71da6f40`](outputs/runs/20261025T181532Z_exercise1_71da6f40/).

Mô hình: `distilbert/distilbert-base-uncased-finetuned-sst-2-english`, đã fine-tune trên SST-2.

Câu `I love this wonderful movie!` được tokenize thành:

```text
[CLS] i love this wonderful movie ! [SEP]
101 1045 2293 2023 6919 3185 999 102
```

Mask toàn 1 vì không có padding trong câu này.

| Câu mẫu | Nhãn | Confidence xấp xỉ |
|---|---|---:|
| I love this wonderful movie! | POSITIVE | 99,99% |
| This movie was terrible and boring. | NEGATIVE | 99,98% |
| The acting was good, but the story was disappointing. | NEGATIVE | 99,67% |

Pipeline và manual forward được kiểm tra trên cùng ba câu với tolerance score `1e-5`. Cả ba khớp. Đây là kiểm chứng nhất quán API, không phải đo accuracy tổng quát của mô hình sentiment.

### 2.2. Exercise 2 — Quá trình fine-tuning

Run hiện hành: [`20261025T181550Z_finetuning_e02bced7`](outputs/runs/20261025T181550Z_finetuning_e02bced7/). Mỗi run có config, environment snapshot, log, provenance, status và SHA256 manifest riêng.

| Epoch | Bước | Loss tối ưu trong epoch | Train accuracy (eval mode) | Validation loss | Validation accuracy |
|---:|---:|---:|---:|---:|---:|
| 1 | 534 | 0,4609 | 88,08% | 0,4036 | 82,27% |
| 2 | 1.068 | 0,2597 | 95,98% | 0,3787 | 84,15% |
| 3 | 1.602 | 0,1515 | 97,21% | 0,4896 | 84,33% |

Toàn bộ train hoàn tất **1.602 optimizer steps**, thời gian Trainer ghi **95,9321 giây**, throughput khoảng 266,751 training samples/s. Không lấy throughput evaluate train để mô tả tốc độ huấn luyện.

Train loss khi evaluate cuối từng epoch là 0,2939 / 0,1261 / 0,0935; không đồng nhất với loss tối ưu trung bình trong epoch vì model mode và thời điểm đo khác nhau.

Epoch 3 có validation accuracy cao nhất nên được chọn đúng tiêu chí. Epoch 2 có validation loss thấp nhất. Không đổi tiêu chí chọn mô hình sau khi đã quan sát kết quả test.

### 2.3. Hệ thống biểu đồ

Các hình dưới là artifacts gốc đã được đối chiếu SHA256, không vẽ lại từ số liệu suy đoán.

![Learning curves](outputs/runs/20261025T181550Z_finetuning_e02bced7/learning_curves.png)

![Confusion matrix](outputs/runs/20261025T181550Z_finetuning_e02bced7/confusion_matrix.png)

![Binary test metrics](outputs/runs/20261025T181550Z_finetuning_e02bced7/metrics_bar.png)

![Train token lengths before truncation](outputs/runs/20261025T181550Z_finetuning_e02bced7/token_length_distribution.png)

![Selected correct and incorrect predictions](outputs/runs/20261025T181550Z_finetuning_e02bced7/sample_predictions.png)

### 2.4. Kiểm chứng độc lập sau khi tiếp nhận output

Lệnh `python Lab3/scripts/audit_outputs.py` thực hiện kiểm tra đọc-only trên artifacts và ghi kết quả vào [`validation/artifact_audit.json`](validation/artifact_audit.json):

- SHA256 khớp với **24 tệp hiện có**: sáu của Exercise 1, 18 của fine-tuning.
- 29 tệp vắng mặt chỉ thuộc `best_model/` và checkpoint trung gian đã chủ động bỏ khỏi Git. Giữ nguyên manifest gốc để lưu dấu vết; không coi các tệp thiếu là đã kiểm tra hash tại máy này.
- Kiểm tra 1.066 index test liên tiếp, 533 nhãn mỗi lớp, probability vectors hợp lệ, predictions khớp argmax.
- Tính lại Accuracy và binary/macro Precision/Recall/F1 từ toàn bộ dự đoán, khớp JSON gốc với tolerance `1e-12`.
- Đối chiếu các sample với bản ghi đầy đủ và kiểm tra lịch sử đủ ba epochs.
- `reload_check.json` ghi nhận trọng số khớp chính xác và suy luận CPU FP32 trên Kaggle. Máy rà soát chỉ đọc bằng chứng này; không chạy checkpoint vì weights không có trong Git.

Ba notebook được thực thi toàn bộ ở `MODE="view"` với Hugging Face offline; kết quả cell và hình được lưu vào notebook. Chế độ view không tải dataset, model hoặc train lại. Bằng chứng thực thi nằm tại `validation/notebook_execution.json`.

## 3. Kết quả, phân tích lỗi và kết luận

### 3.1. Kết quả test của checkpoint đã chọn

| Metric | Giá trị |
|---|---:|
| Accuracy | **83,9587%** |
| Binary Precision | 83,5185% |
| Binary Recall | 84,6154% |
| Binary F1 | **84,0634%** |
| Macro Precision | 83,9646% |
| Macro Recall | 83,9587% |
| Macro F1 | 83,9580% |

Confusion matrix theo thứ tự `[Negative, Positive]`:

| Nhãn thật / dự đoán | Negative | Positive |
|---|---:|---:|
| Negative | TN = 444 | FP = 89 |
| Positive | FN = 82 | TP = 451 |

Đúng **895**, sai **171** trên 1.066 mẫu. Recall Negative là 444/533 = 83,30%; Recall Positive là 451/533 = 84,62%. Mô hình cao hơn baseline hằng 50% khoảng **33,96 điểm phần trăm accuracy**, nhưng chưa có nhiều seed để kết luận về biến thiên hoặc ý nghĩa thống kê.

### 3.2. Phân tích khả năng tổng quát hóa

Train accuracy tăng đến 97,21% trong khi validation accuracy 84,33%, chênh khoảng 12,88 điểm phần trăm. Validation loss tăng mạnh ở epoch 3 trong khi loss train tiếp tục giảm: dấu hiệu overfitting. Accuracy validation chỉ tăng 2/1.066 mẫu từ epoch 2 đến epoch 3, nên không diễn giải cải thiện đó là lớn.

Nếu thực hiện thí nghiệm tiếp theo, có thể đánh giá early stopping hoặc regularization bằng validation và ngân sách cố định trước; không dùng test để tiếp tục chọn siêu tham số. Kết quả hiện hành được giữ nguyên, không loại run vì chưa đạt một mốc tự đặt.

### 3.3. Phân tích định tính các lỗi thực

Các mẫu trong `sample_predictions.json` ưu tiên confidence cao của cả đúng/sai và cả hai nhãn, không phải mẫu ngẫu nhiên đại diện toàn bộ test. Diễn giải sau là giả thuyết ngôn ngữ, chưa phải causal attribution.

**Hey Arnold — Negative bị dự đoán Positive (confidence khoảng 99,6%).** Câu chê “stretched to barely feature length” nhưng kết thúc bằng “story is appealing” và có lời khen hình ảnh. Hai chiều cảm xúc có thể khiến mô hình thiên về các tín hiệu khen cuối câu. Không chỉ gắn nhãn chung chung là lỗi phủ định: cấu trúc trộn lời chê và lời khen mới là điểm nổi bật.

**Disney — Negative bị dự đoán Positive (khoảng 99,6%).** Câu “entertains not so much … but through the perverse pleasure … scrape the bottom of its own cracker barrel” dùng mỉa mai. “Entertains” có vẻ tích cực, nhưng thành ngữ và ngữ cảnh chê việc Disney cạn ý tưởng mới quyết định sắc thái chung. Phân loại dựa vào từ cảm xúc đơn lẻ có thể thất bại ở cấu trúc này.

**“story is almost an afterthought” — Positive bị dự đoán Negative (khoảng 99,6%).** “Afterthought” và “inexplicable” thường mang nghĩa chê, trong khi câu có thể mô tả trải nghiệm thiên về màu sắc được dataset xem tích cực. Nhãn thật được giữ như dataset cung cấp; không khẳng định nhãn sai chỉ vì con người cũng có thể thấy câu mơ hồ.

**The Wild Thornberrys — Positive bị dự đoán Negative (khoảng 99,6%).** “Sibling rivalry” và “family chaos” nói về nội dung xung đột gia đình. Cụm “to which anyone can relate” lại thể hiện tính gần gũi và đồng cảm tích cực. Mô hình có thể nhầm từ vựng mô tả cốt truyện với thái độ đánh giá.

Một số dự đoán sai vẫn có confidence cao. Điều này chứng minh confidence không bảo đảm đúng; cần calibration curve/ECE trên tập phù hợp để kết luận đầy đủ về hiệu chuẩn, chưa được thực hiện trong lab này.

### 3.4. Kết luận và khả năng sử dụng

Bài thực hiện đủ hai Exercise, có mô hình pretrained cho inference, full official dataset, tokenization, TrainingArguments/Trainer, fine-tuning và đánh giá test. Các kết quả có run ID và artifacts đối chiếu được; ba notebook có thể mở xem ngay mà không cần train lại.

**Giới hạn lưu trữ:** GitHub không chứa checkpoint, không có URL checkpoint công khai đã xác minh. Người thực hiện cần giữ ZIP Kaggle `lab03_results_20261025T181550Z_finetuning_e02bced7.zip`. Khôi phục `best_model/` đúng run để đánh giá/suy luận lại. SHA256 trọng số kỳ vọng: `5a63c9c2a6391132c86b0e6c08277cc23ccc692dd080b78ba2db074225bb5105`.

Xem [README](README.md) cho cài đặt, train/view và lệnh eval/infer. Nhật ký sửa lỗi nội bộ nằm ở [internal_changes.md](internal_changes.md), không phải nội dung báo cáo thực nghiệm.
