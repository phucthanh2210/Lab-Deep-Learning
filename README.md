# Lab Deep Learning

## Lab 03 — Getting started with Hugging Face

Bài thực hành đã có mã nguồn, ba notebook có output và báo cáo dựa trên kết quả Kaggle:

- **Exercise 1:** pretrained sentiment analysis, tokenization và đối chiếu pipeline/manual forward.
- **Exercise 2:** fine-tune DistilBERT trên toàn bộ Rotten Tomatoes: **8.530 train / 1.066 validation / 1.066 test**.
- **Test:** Accuracy **83,96%**, Binary F1 **84,06%**, đúng 895/1.066 mẫu.
- **Kiểm chứng:** 24 artifacts khớp SHA256; metrics được tính lại từ toàn bộ predictions; 13 tests CPU đạt; ba notebook thực thi chế độ view offline.

### Mở bài

- [Hướng dẫn Lab3](Lab3/README.md)
- [Notebook tích hợp có output](Lab3/notebooks/lab03_huggingface_complete.ipynb)
- [Báo cáo thực hành](Lab3/walkthrough_lab_03.md)
- [Audit kết quả](Lab3/validation/artifact_audit.json)

Notebook mặc định `MODE="view"` để xem kết quả mà không train lại. Muốn thực hiện thí nghiệm mới, chọn `MODE="train"` trên máy GPU theo README. Trọng số không nằm trong Git; cần khôi phục từ ZIP Kaggle riêng để chạy inference/eval với checkpoint đã train.
