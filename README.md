# Phân tích Cảm xúc

## Giới thiệu

Phân tích cảm xúc là một lĩnh vực quan trọng trong xử lý ngôn ngữ tự nhiên, cho phép hiểu và đánh giá cảm xúc từ dữ liệu văn bản.  
Sentiment analysis (phân tích cảm xúc) là quá trình sử dụng các kỹ thuật xử lý ngôn ngữ tự nhiên (NLP), học máy hoặc thống kê để xác định và phân loại cảm xúc, thái độ hoặc ý kiến được thể hiện trong một đoạn văn bản, chẳng hạn như tích cực, tiêu cực hoặc trung lập.

## Mục đích và Ứng dụng

### Mục đích
Hiểu được cảm xúc hoặc ý kiến của người dùng từ dữ liệu văn bản, như bài đánh giá, bình luận trên mạng xã hội, hoặc phản hồi khách hàng.

### Ứng dụng
- **Kinh doanh**: Phân tích phản hồi khách hàng để cải thiện sản phẩm/dịch vụ.
- **Mạng xã hội**: Theo dõi xu hướng cảm xúc về thương hiệu, sự kiện hoặc sản phẩm.
- **Chính trị**: Đánh giá dư luận về các chính sách hoặc ứng cử viên.
- **Y tế**: Phát hiện cảm xúc trong phản hồi của bệnh nhân để cải thiện chăm sóc.

## Phương pháp luận

### 2.1. Mục tiêu
Xây dựng hệ thống phân tích cảm xúc tự động để phân loại văn bản (đánh giá, bình luận mạng xã hội) thành tích cực, tiêu cực hoặc trung lập, sử dụng các phương pháp thống kê, học máy và mô hình ngôn ngữ lớn (LLMs).

### 2.2. Thu thập và Chuẩn bị Dữ liệu
- **Nguồn dữ liệu**: Sử dụng tập dữ liệu công khai (Amazon, Twitter, IMDb) hoặc dữ liệu thu thập từ mạng xã hội qua Tweepy/BeautifulSoup.
- **Gắn nhãn**: Đảm bảo dữ liệu có nhãn cảm xúc (tích cực, tiêu cực, trung lập).
- **Làm sạch**: Loại bỏ trùng lặp, văn bản không liên quan, chuẩn hóa định dạng.

### 2.3. Tiền Xử Lý và Trích Xuất Đặc Trưng
- **Tiền xử lý**: Chuyển văn bản về chữ thường, xóa dấu câu, từ dừng, ký tự đặc biệt; áp dụng tokenization và lemmatization/stemming.
- **Trích xuất đặc trưng**:
  - **TF-IDF**: Biểu diễn văn bản bằng vector số, ưu tiên các từ hiếm và quan trọng.
  - **N-grams**: Sử dụng unigram, bigram, trigram để nắm bắt ngữ cảnh (ví dụ: "rất tốt", "không hài lòng").
  - **Word Embeddings**: Áp dụng nhúng từ (Word2Vec, GloVe) hoặc nhúng ngữ cảnh từ mô hình Transformer (BERT, Gemma 2).

### 2.4. Phương Pháp Phân Loại

#### a. Phương pháp Thống kê
- **VADER**:
  - Mô hình: Sử dụng VADER, từ điển tối ưu cho văn bản mạng xã hội, gán điểm cảm xúc tổng hợp (-1 đến 1).
  - Phân loại: Điểm ≥ 0.05 (tích cực), ≤ -0.05 (tiêu cực), còn lại (trung lập).
  - Đánh giá: So sánh dự đoán với nhãn thực để tính độ chính xác.

- **Markov Chain**:
  - Tiền xử lý & cân bằng: Làm sạch văn bản và lấy 500 mẫu mỗi lớp.
  - Huấn luyện: Tạo mô hình Markov (bigram + unigram) cho từng nhãn sentiment.
  - Dự đoán: Tính xác suất chuỗi và gán nhãn có xác suất cao nhất.

#### b. Học Máy (ML)
- **Trích xuất đặc trưng**: Chuyển văn bản thành vector bằng TF-IDF, kết hợp N-grams.
- **Xử lý dữ liệu không cân bằng**: Sử dụng RandomOverSampler để cân bằng lớp thiểu số.
- **Mô hình**:
  - **Multinomial Naive Bayes**: Phù hợp với văn bản, huấn luyện nhanh.
  - **Logistic Regression**: Hiệu quả với đặc trưng TF-IDF.
  - **SVM**: Tốt cho biên phân loại rõ ràng.
  - **Decision Tree/Random Forest**: Khắc phục overfitting, cải thiện độ chính xác.
  - **KNN**: Phân loại dựa trên khoảng cách, phù hợp với dữ liệu có cụm.

#### c. Mô hình Ngôn ngữ Lớn (LLMs)
- **Mô hình**: Sử dụng Gemma 2 (2.6 tỷ tham số, kiến trúc Transformer) với phiên bản gemma2_instruct_2b_en, triển khai trên JAX, keras-nlp, tensorflow.
- **Quy trình**:
  - **Prompt Engineering**: Định dạng lời nhắc: “You are an AI expert in sentiment analysis. Return one label: Positive, Negative, or Neutral.”
  - **Batch Processing**: Xử lý lô 32 mẫu, giới hạn 256 token.
  - **Fine-tuning**: Tinh chỉnh bằng LoRA (lora_rank=4, lr=1×10⁻⁴, 1 epoch) trên 1,000 mẫu huấn luyện.
  - **Cấu hình**: Tối ưu bộ nhớ với XLA_PYTHON_CLIENT_MEM_FRACTION=1.00.

#### d. Đánh giá mô hình (Model Evaluation)
Hiệu suất mô hình được đo bằng các chỉ số:
- **Accuracy**: Tỷ lệ dự đoán đúng trên tổng số mẫu.
- **Precision, Recall, F1-Score**: Đặc biệt quan trọng trong việc đánh giá các lớp không cân bằng (như lớp tiêu cực rất ít).
- **Confusion Matrix**: Ma trận nhầm lẫn cho thấy chi tiết các dự đoán đúng/sai của từng lớp.
