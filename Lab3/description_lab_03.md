# Practice 3 — Get started with Hugging Face

## Đề gốc từ ảnh giảng viên

### Exercise 1: Sentiment Analysis with Hugging Face
1. Install the Hugging Face transformers library.
2. Use a pre-trained sentiment analysis model from the Hugging Face Hub.
3. Tokenize a sample sentence.
4. Perform sentiment analysis on the sentence.

### Exercise 2: Finetuning a Pretrained Model for Binary Text Classification
1. Install the necessary Hugging Face libraries (transformers, datasets, evaluate).
2. Load a simple dataset for binary text classification.
3. Load a pretrained model and its tokenizer.
4. Preprocess the dataset to be suitable for the model.
5. Define training arguments.
6. Create a Trainer object and finetune the model.
7. Evaluate the finetuned model.

## Yêu cầu bổ sung của người dùng

Dùng toàn bộ Rotten Tomatoes (8.530 train, 1.066 validation, 1.066 test), ba notebook chuyên biệt/tổng hợp, isolated runs và registry JSON, năm biểu đồ, phân tích lỗi định tính, tests đối chiếu pipeline/manual và sanity overfit dùng riêng trong tests. Đây là phần mở rộng do người dùng yêu cầu, không phải ngưỡng đánh giá có trong đề gốc.

Đề gốc không yêu cầu Accuracy tối thiểu 85%, không chỉ định GPU hoặc dataset cụ thể. Báo cáo phải dùng kết quả chạy thực tế, không sao chép số liệu từ bản tham khảo.

## Tài liệu chính thức

- https://huggingface.co/docs/transformers/en/training
- https://huggingface.co/blog/sentiment-analysis-python
- https://huggingface.co/datasets/cornell-movie-review-data/rotten_tomatoes
