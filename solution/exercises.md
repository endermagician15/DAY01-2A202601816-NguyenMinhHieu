# K4 — Ngày 1: Bài Tập & Phản Ánh
## Khám Phá LLM API | Phiếu Thực Hành

**Thời lượng:** 14h00–18h00
**Cách làm:** Trả lời từng câu ngay sau khi hoàn thành block tương ứng —
đừng để dồn hết về cuối buổi. Thay dòng `*Câu trả lời của bạn*` bằng câu
trả lời thật (chấm tự động sẽ đếm số câu đã trả lời).

---

## Block 1 — API Cơ Bản (trả lời sau Checkpoint 1)

### Câu 1.1 — Độ nhạy của temperature
Gọi `call_openai` với temperature 0.0, 0.7, 1.2 và 1.8 dùng prompt
**"Hãy kể cho tôi một sự thật thú vị về Hà Nội."**

**Bạn nhận thấy quy luật gì qua bốn phản hồi? Ở mức nào phản hồi bắt đầu
kém mạch lạc?** (2–3 câu)
> Temperature càng tăng thì response càng sáng tạo. Output đi từ kể về các phố cổ của Hà Nội ở 0.0, đến kể về truyền thuyết hồ gươm ở 1.2. Tuy nhiên model hallucinate nặng ở temperature 1.8 và output ra lẫn lộn các ngôn ngữ

### Câu 1.2 — Chọn temperature cho sản phẩm
**Bạn sẽ đặt temperature bao nhiêu cho trợ lý soạn thảo hợp đồng pháp lý,
và bao nhiêu cho trợ lý viết slogan quảng cáo? Giải thích khác biệt.**
> Đối với trợ lý soạn thảo hợp đồng pháp lý, temperature 0.0 - 0.2 sẽ là hợp lý nhất nhằm đảm bảo độ chính xác. Còn đối với trợ lý viết slogan quảng cáo thì temperature 0.7 - 1.0 là hợp lý để có sự sáng tạo, vừa đảm bảo model không lỗi hoàn toàn

### Câu 1.3 — Đánh đổi chi phí
Kịch bản: 20.000 người dùng hoạt động mỗi ngày, mỗi người gọi API 2 lần,
mỗi lần trung bình ~500 token đầu ra.

**Ước tính chi phí mỗi ngày của model lớn so với model nhỏ cho workload này
(dựa trên bảng giá trong template). Nêu một trường hợp model lớn xứng đáng
với chi phí và một trường hợp model nhỏ là lựa chọn đúng:**
> GPT-4o: 20000 * 2 * 500 * 0.01 / 1000 * 0.75 = 200 USD
> GPT-4o-mini: 20000 * 2 * 500 * 0.0006 / 1000 * 0.75 = 12 USD
> Trường hợp model lớn xứng đáng với chi phí: chatbot hỗ trợ pháp lý, yêu cầu độ chính xác cao và không cho phép sai sót.
> Trường hợp model nhỏ là lựa chọn đúng: chatbot trả lời câu hỏi thường gặp hoặc cskh

---

## Block 2 — System Prompt & Token (trả lời sau Checkpoint 2)

### Câu 2.1 — Sức mạnh của persona
Gọi `chat_with_system_prompt` hai lần với cùng câu hỏi
**"Giải thích máy học (machine learning) là gì?"** nhưng hai system prompt
khác nhau:
- "Bạn là một nhà thơ, trả lời mọi thứ bằng hình ảnh ví von, tránh thuật ngữ."
- "Bạn là kỹ sư phần mềm senior, trả lời chính xác, có ví dụ code khi phù hợp."

**Hai phản hồi khác nhau như thế nào (giọng văn, độ dài, mức kỹ thuật)?
Từ đó rút ra system prompt điều khiển được những khía cạnh nào của phản hồi?**
(3–4 câu)
> System prompt cho phép điều chỉnh giọng văn, format, phạm vi từ vựng,.. của output. Output của nhà thơ ngắn hơn, không có độ sâu kỹ thuật trong khi output của kỹ sư nặng thuật ngữ và nhiều giá trị kỹ thuật

### Câu 2.2 — tiktoken vs đếm từ
Chọn một đoạn văn tiếng Việt ~150 từ. So sánh số token theo `count_tokens`
(tiktoken) với ước lượng `số từ / 0.75` mà Part 1 đã dùng.

**Hai con số chênh nhau bao nhiêu phần trăm? Nếu dùng ước lượng thô để dự
toán ngân sách API cho ứng dụng tiếng Việt, bạn sẽ dự toán thiếu hay thừa —
và vì sao?**
> Tiktoken đếm nhiều hơn so với ước lượng thô khoảng 30% và nếu sử dụng ước lượng thô cho tiếng việt thì kết quả sẽ luôn thiếu hụt do tiếng việt có dấu

---

## Block 3 — Streaming & Độ Bền (trả lời sau Checkpoint 3)

### Câu 3.1 — Trải nghiệm người dùng với streaming
**Xét ba ứng dụng: (a) chatbot văn bản, (b) trợ lý giọng nói đọc to phản hồi,
(c) pipeline dịch tài liệu chạy ngầm ban đêm. Ứng dụng nào hưởng lợi nhiều
nhất từ streaming, ứng dụng nào không cần — và tại sao?** (1 đoạn văn)
> Pipeline dịch tài liệu chạy ngầm sẽ không cần đến streaming, trong khi đó chatbot văn bản và trợ lý giọng nói cần đến streaming để tương tác với người dùng theo thời gian thực

### Câu 3.2 — Vì sao backoff theo cấp số nhân?
**Khi API quá tải và hàng nghìn client cùng retry, exponential backoff giúp
gì so với delay cố định? Tra cứu thêm: kỹ thuật "jitter" (thêm độ trễ ngẫu
nhiên) giải quyết vấn đề gì còn sót lại?**
> Nếu tất cả client cùng retry theo delay cố định thì hệ thống sẽ liên tục nhận các đợt retry dẫn đến không thể phục hồi. Exponential backoff giúp hệ thống có khoảng trống đủ dài để giải quyết dần hàng đợi. Tuy nhiên, exponential backoff vẫn sẽ gặp vấn đề nếu các request đến cùng thời điểm và nhận khoảng chờ giống nhau. Jitter giải quyết vấn đề này bằng việc thêm các độ trễ ngẫu nhiên

---

## Block 4 — Mini-Project (trả lời sau Checkpoint 4)

### Câu 4.1 — Thiết kế persona
**Viết lại system prompt bạn dùng cho trợ lý của mình. Chỉ ra 2 chỗ trong
prompt mà nếu xóa đi, hành vi trợ lý sẽ thay đổi rõ rệt — và mô tả thay đổi
đó:**
> System prompt: "Bạn là trợ lý tư vấn lập trình Python. Hãy giải thích ngắn gọn bản chất vấn đề và chỉ đưa ra cú pháp/hướng dẫn. (1) Không tạo đoạn mã (code mẫu) hoàn chỉnh. (2) Trả lời hoàn toàn bằng tiếng Việt."
> - Nếu xóa (1): Agent sẽ bắt đầu sinh code mẫu ví dụ vào response
> - Nếu xóa (2): Agent sẽ trả lời bằng tiếng Anh khi prompt là tiếng Anh

### Câu 4.2 — Hạn chế & cải thiện
**Trợ lý của bạn giữ history 4 lượt cuối. Hãy mô tả một tình huống hội thoại
cụ thể mà giới hạn này khiến trợ lý trả lời sai/mất ngữ cảnh, và đề xuất một
cách khắc phục (ví dụ: tóm tắt các lượt cũ, tăng giới hạn có chọn lọc...):**
> Giới hạn 4 lượt cuối khiến Model mất ngữ cảnh trong các phiên hội thoại dài. Để khắc phục nhược điểm này có thể tóm tắt các lượt cũ bằng model mini để vừa tối ưu chi phí vừa không mất đi ngữ cảnh hội thoại

---

## Danh Sách Kiểm Tra Nộp Bài

- [ ] `python grade.py` — xem điểm tự động, mục tiêu ≥ 75/100
- [ ] Cả 4 checkpoint pytest đều pass
- [ ] Tất cả 9 câu trong file này đã được trả lời
- [ ] Đã copy bài làm vào folder `solution/`, push lên GitHub cá nhân và nộp link repo vào vlearn (theo hướng dẫn README)
