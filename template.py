"""
K4 — Ngày 1: Khám Phá LLM API (14h00–18h00)
AICB-P1: AI Practical Competency Program, Phase 1

Hướng dẫn:
    1. Làm theo LAB_GUIDE.md — mỗi block có các bước chi tiết và checkpoint.
    2. Điền vào tất cả các chỗ đánh dấu TODO.
    3. KHÔNG đổi chữ ký hàm (tên hàm, tham số).
    4. Import OpenAI BÊN TRONG hàm (xem gợi ý) — nếu import ở đầu file,
       các bài test mock sẽ không hoạt động.
    5. Kiểm tra tiến độ:  pytest tests/test_part1.py -v  (từng phần)
       Chấm điểm tổng:    python grade.py
"""

import os
import time
from typing import Any, Callable

from dotenv import load_dotenv

# Nạp OPENAI_API_KEY từ file .env (copy .env.example thành .env và dán key vào)
load_dotenv()

# ---------------------------------------------------------------------------
# Bảng giá ước tính (USD / 1K token) — cập nhật nếu giá thay đổi
# ---------------------------------------------------------------------------
PRICING_PER_1K_TOKENS = {
    "gpt-4o": {"input": 0.0025, "output": 0.010},
    "gpt-4o-mini": {"input": 0.00015, "output": 0.0006},
    "gemini-2.5-flash": {"input": 0.0003, "output": 0.0025},
    "gemini-2.5-flash-lite": {"input": 0.0001, "output": 0.0004},
}

# Luồng chính: OpenAI (mặc định, không cần đặt gì trong .env).
# Không có key OpenAI? Dùng luồng thay thế Google Gemini (Phụ lục B
# trong LAB_GUIDE.md) — tên model đổi qua .env. NVIDIA NIM: Phụ lục C.
OPENAI_MODEL = os.getenv("LAB_MODEL", "gpt-4o")
OPENAI_MINI_MODEL = os.getenv("LAB_MINI_MODEL", "gpt-4o-mini")


# ===========================================================================
# PART 1 — API CƠ BẢN (Block 1: 15h00–15h40)
# ===========================================================================

# ---------------------------------------------------------------------------
# Task 1.1 — Gọi GPT-4o
# ---------------------------------------------------------------------------
def call_openai(
    prompt: str,
    model: str = OPENAI_MODEL,
    temperature: float = 0.7,
    top_p: float = 0.9,
    max_tokens: int = 256,
) -> tuple[str, float]:
    """
    Gọi OpenAI Chat Completions API, trả về nội dung phản hồi + độ trễ.

    Args:
        prompt:      Tin nhắn của người dùng.
        model:       Model OpenAI sử dụng (mặc định: gpt-4o).
        temperature: Độ ngẫu nhiên khi lấy mẫu (0.0 – 2.0).
        top_p:       Ngưỡng nucleus sampling.
        max_tokens:  Số token tối đa được sinh ra.

    Returns:
        Tuple (response_text: str, latency_seconds: float).

    Gợi ý:
        from openai import OpenAI            # import BÊN TRONG hàm
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        # đo thời gian bằng time.time() trước và sau lời gọi API
    """
    from openai import OpenAI
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    start_time = time.perf_counter()
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role":"user", "content":prompt}
        ],
        max_tokens=max_tokens,
        temperature=temperature,
    )
    end_time = time.perf_counter()
    latency_seconds = end_time - start_time
    response_text = response.choices[0].message.content
    return response_text, latency_seconds


# ---------------------------------------------------------------------------
# Task 1.2 — Gọi GPT-4o-mini
# ---------------------------------------------------------------------------
def call_openai_mini(
    prompt: str,
    temperature: float = 0.7,
    top_p: float = 0.9,
    max_tokens: int = 256,
) -> tuple[str, float]:
    """
    Gọi API với model gpt-4o-mini — nhanh hơn và rẻ hơn.

    Returns:
        Tuple (response_text: str, latency_seconds: float).

    Gợi ý:
        Tái sử dụng call_openai() với model=OPENAI_MINI_MODEL — 1 dòng code.
    """
    return call_openai(
        prompt=prompt,
        model=OPENAI_MINI_MODEL,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
    )


# ---------------------------------------------------------------------------
# Task 1.3 — So sánh GPT-4o vs GPT-4o-mini
# ---------------------------------------------------------------------------
def compare_models(prompt: str) -> dict:
    """
    Gọi cả hai model với cùng một prompt và trả về dict so sánh.

    Returns:
        Dict với các key:
            - "gpt4o_answer":      str
            - "mini_answer":       str
            - "gpt4o_time":       float
            - "mini_time":        float
            - "gpt4o_cost": float  (USD ước tính cho phản hồi)

    Gợi ý:
        pricing = PRICING_PER_1K_TOKENS.get(
            OPENAI_MODEL, PRICING_PER_1K_TOKENS["gpt-4o"]
        )
        cost = (len(response.split()) / 0.75) / 1000 * pricing["output"]
        (0.75 từ ≈ 1 token — ước lượng thô; Part 2 sẽ tính chính xác hơn.
         Dùng .get để lấy đúng giá model đang chạy — gpt-4o, gemini...;
         model không có trong bảng thì lấy giá gpt-4o làm tham chiếu)
    """
    gpt4o_answer, gpt4o_time = call_openai(prompt)
    gpt4o_pricing = PRICING_PER_1K_TOKENS.get(OPENAI_MODEL, PRICING_PER_1K_TOKENS["gpt-4o"])
    gpt4o_cost = (len(gpt4o_answer.split()) / 0.75) / 1000 * gpt4o_pricing["output"]
    mini_answer, mini_time = call_openai_mini(prompt)
    mini_pricing = PRICING_PER_1K_TOKENS.get(OPENAI_MINI_MODEL, PRICING_PER_1K_TOKENS["gpt-4o"])
    mini_cost = (len(mini_answer.split()) / 0.75) / 1000 * mini_pricing["output"]
    return {
        "gpt4o_answer": gpt4o_answer,
        "mini_answer": mini_answer,
        "gpt4o_time": gpt4o_time,
        "mini_time": mini_time,
        "gpt4o_cost": gpt4o_cost,
        "mini_cost": mini_cost,
    }


# ===========================================================================
# PART 2 — SYSTEM PROMPT & TOKEN (Block 2: 15h40–16h20)
# ===========================================================================

# ---------------------------------------------------------------------------
# Task 2.1 — Chat với system prompt (persona)
# ---------------------------------------------------------------------------
def chat_with_system_prompt(
    system_prompt: str,
    user_prompt: str,
    model: str = OPENAI_MODEL,
    temperature: float = 0.7,
    max_tokens: int = 256,
) -> tuple[str, float]:
    """
    Gọi API với MESSAGES gồm 2 phần: system prompt (định hình vai trò/persona
    của model) và user prompt (câu hỏi thật).

    Args:
        system_prompt: Chỉ dẫn vai trò, ví dụ "Bạn là giáo viên tiểu học,
                       giải thích mọi thứ thật đơn giản."
        user_prompt:   Tin nhắn của người dùng.

    Returns:
        Tuple (response_text: str, latency_seconds: float).

    Gợi ý:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
    """
    from openai import OpenAI
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    start_time = time.perf_counter()
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role":"system", "content":system_prompt},
            {"role":"user", "content":user_prompt},
        ],
        max_tokens=max_tokens,
        temperature=temperature,
    )
    end_time = time.perf_counter()
    latency_seconds = end_time - start_time
    response_text = response.choices[0].message.content
    return response_text, latency_seconds


# ---------------------------------------------------------------------------
# Task 2.2 — Đếm token bằng tiktoken
# ---------------------------------------------------------------------------
def count_tokens(text: str, model: str = OPENAI_MODEL) -> int:
    """
    Đếm số token của một đoạn text bằng thư viện tiktoken.

    Args:
        text:  Đoạn text cần đếm.
        model: Model dùng để chọn bộ mã hóa (encoding).

    Returns:
        Số token (int).

    Gợi ý:
        import tiktoken
        enc = tiktoken.encoding_for_model(model)
        return len(enc.encode(text))

        tiktoken cần tải bộ mã hóa từ mạng ở lần chạy đầu. Hãy bọc trong
        try/except — nếu lỗi (offline, model lạ), dùng ước lượng dự phòng:
        max(1, len(text) // 4)   (trung bình 1 token ≈ 4 ký tự)
    """
    import tiktoken
    try:
        enc = tiktoken.encoding_for_model(model)
        return len(enc.encode(text))
    except:
        return max(1, len(text) // 4)

# ---------------------------------------------------------------------------
# Task 2.3 — Ước tính chi phí chính xác
# ---------------------------------------------------------------------------
def estimate_cost(prompt: str, response: str, model: str = OPENAI_MODEL) -> dict:
    """
    Tính chi phí một lượt gọi API dựa trên số token THẬT (đếm bằng
    count_tokens) và bảng giá PRICING_PER_1K_TOKENS — tách riêng chi phí
    input (prompt) và output (response).

    Returns:
        Dict với các key:
            - "prompt_tokens":  int
            - "completion_tokens": int
            - "prompt_cost":    float  (USD)
            - "completion_cost":   float  (USD)
            - "total_cost":    float  (USD)

    Gợi ý:
        pricing = PRICING_PER_1K_TOKENS.get(model, PRICING_PER_1K_TOKENS["gpt-4o"])
        prompt_cost = prompt_tokens / 1000 * pricing["input"]
        (.get với fallback: model không có trong bảng giá — ví dụ model NIM
         miễn phí — thì lấy giá gpt-4o làm tham chiếu học tập)
    """
    prompt_tokens = count_tokens(prompt, model)
    commpletion_tokens = count_tokens(response, model)
    pricing = PRICING_PER_1K_TOKENS.get(model, PRICING_PER_1K_TOKENS["gpt-4o"])
    prompt_cost = prompt_tokens / 1000 * pricing["input"]
    completion_cost = commpletion_tokens / 1000 * pricing["output"]
    return{
        "prompt_tokens": prompt_tokens,
        "completion_tokens": commpletion_tokens,
        "prompt_cost": prompt_cost,
        "completion_cost": completion_cost,
        "total_cost": prompt_cost + completion_cost,
    }


# ===========================================================================
# PART 3 — STREAMING & ĐỘ BỀN (Block 3: 16h30–17h10)
# ===========================================================================

# ---------------------------------------------------------------------------
# Task 3.1 — Chatbot streaming có lịch sử hội thoại
# ---------------------------------------------------------------------------
def streaming_chatbot() -> None:
    """
    Chatbot dòng lệnh tương tác dùng streaming.

    Hành vi:
        - Stream token từ OpenAI ngay khi chúng được sinh ra (in từng chunk).
        - Duy trì 4 lượt hội thoại gần nhất trong history.
        - Gõ 'quit', 'exit' hoặc 'bye' để thoát.

    Gợi ý:
        - Giữ list `history` gồm các dict {"role": ..., "content": ...}.
        - Dùng stream=True trong client.chat.completions.create() và lặp:
            for chunk in stream:
                delta = chunk.choices[0].delta.content or ""
                print(delta, end="", flush=True)
        - Sau mỗi lượt, thêm phản hồi assistant vào history.
        - Cắt history còn 4 lượt cuối (8 message): history = history[-8:]
    """
    from openai import OpenAI
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    history = []
    while True:
        user_input = input("User: ")
        if user_input.lower() in ['quit','exit','bye']:
            break
        history.append({"role":"user","content":user_input})
        stream = client.chat.completions.create(
            model="gpt-4o",
            messages=history,
            stream=True,
        )
        response_text = ""
        print("AI: ", end="", flush=True)
        for chunk in stream:
            delta = chunk.choices[0].delta.content or ""
            print(delta, end="", flush=True)
            response_text += delta
        print()
        history.append({"role":"assistant","content":response_text})
        history = history[-8:]


# ---------------------------------------------------------------------------
# Task 3.2 — Retry với exponential backoff
# ---------------------------------------------------------------------------
def retry_with_backoff(
    fn: Callable,
    max_retries: int = 3,
    base_delay: float = 0.1,
) -> Any:
    """
    Gọi fn(). Nếu ném exception, thử lại tối đa max_retries lần với
    exponential backoff (delay = base_delay * 2^attempt).

    Args:
        fn:          Callable không tham số.
        max_retries: Số lần thử lại tối đa.
        base_delay:  Delay ban đầu (giây) trước lần thử lại đầu tiên.

    Returns:
        Giá trị trả về của fn() khi thành công.

    Raises:
        Exception cuối cùng của fn() sau khi hết số lần thử.
    """
    retries_left = max_retries
    delay = base_delay

    while retries_left > 0:
        try:
            return fn()
        except Exception as e:
            retries_left -= 1
            if retries_left == 0:
                raise e
            print(f"Lỗi: {e} — thử lại sau {delay:.2f}s")
            time.sleep(delay)
            delay *= 2
    


# ===========================================================================
# PART 4 — MINI-PROJECT: TRỢ LÝ CLI HOÀN CHỈNH (Block 4: 17h10–17h50)
# ===========================================================================
def run_assistant(
    persona: str,
    get_input: Callable[[], str] = None,
    max_turns: int = None,
) -> dict:
    """
    Trợ lý CLI hoàn chỉnh — ghép mọi thứ bạn đã xây trong Part 1–3.

    Hành vi:
        1. Dùng `persona` làm system prompt cho TOÀN BỘ phiên chat.
        2. Mỗi lượt: đọc tin nhắn qua get_input(); nếu là 'quit'/'exit'/'bye'
           (không phân biệt hoa thường) → kết thúc phiên.
        3. Gọi API với stream=True, messages = system + history + tin nhắn mới.
           Bọc lời gọi API trong retry_with_backoff để chịu lỗi tạm thời.
        4. In từng chunk khi stream về, ghép lại thành reply hoàn chỉnh.
        5. Cập nhật history (user + assistant), giữ tối đa 4 lượt cuối
           (8 message): history = history[-8:]
        6. Cộng dồn thống kê bằng count_tokens và estimate_cost.
        7. Dừng khi đạt max_turns (nếu được đặt).

    Args:
        persona:   Mô tả vai trò, dùng làm system prompt.
        get_input: Hàm đọc input (mặc định: input). Tham số này giúp
                   test tự động không cần bàn phím thật.
        max_turns: Số lượt tối đa (None = không giới hạn).

    Returns:
        Dict thống kê phiên chat:
            - "turns":    int   (số lượt hỏi–đáp đã thực hiện)
            - "tokens_used": int   (tổng token user + assistant)
            - "total_cost":   float (tổng USD ước tính)
            - "history":      list  (history còn lại sau khi cắt, ≤ 8 message)

    Gợi ý khung sườn:
        if get_input is None:
            get_input = input
        history, turns, tokens_used, total_cost = [], 0, 0, 0.0
        while True:
            if max_turns is not None and turns >= max_turns:
                break
            user_msg = get_input()
            if user_msg.strip().lower() in ("quit", "exit", "bye"):
                break
            messages = [{"role": "system", "content": persona}] + history \\
                       + [{"role": "user", "content": user_msg}]
            # stream = retry_with_backoff(lambda: client.chat...create(
            #              model=..., messages=messages, stream=True))
            # reply = ghép các chunk...
            ...
        return {"turns": turns, "tokens_used": tokens_used,
                "total_cost": total_cost, "history": history}
    """

    if get_input is None:
        get_input = input

    history = []
    turns = 0
    tokens_used = 0
    total_cost = 0.0

    from openai import OpenAI

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    while True:
        if max_turns is not None and turns >= max_turns:
            break

        user_msg = get_input()

        if user_msg.strip().lower() in ("quit", "exit", "bye"):
            break

        messages = (
            [{"role": "system", "content": persona}]
            + history
            + [{"role": "user", "content": user_msg}]
        )

        stream = retry_with_backoff(
            lambda: client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=messages,
                stream=True,
            )
        )

        response_text = ""
        print("AI: ", end="", flush=True)

        for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                delta = chunk.choices[0].delta.content
                print(delta, end="", flush=True)
                response_text += delta
        print()

        turns += 1

        history.append({"role": "user", "content": user_msg})
        history.append({"role": "assistant", "content": response_text})
        history = history[-8:]

        cost_info = estimate_cost(
            prompt=user_msg,
            response=response_text,
            model=OPENAI_MODEL,
        )

        total_cost += cost_info["total_cost"]
        tokens_used += (
            cost_info["prompt_tokens"] + cost_info["completion_tokens"]
        )

    return {
        "turns": turns,
        "tokens_used": tokens_used,
        "total_cost": total_cost,
        "history": history,
    }


# ===========================================================================
# BONUS (không bắt buộc — cho bạn nào xong sớm)
# ===========================================================================
def batch_compare(prompts: list[str]) -> list[dict]:
    """
    Chạy compare_models cho từng prompt trong list.

    Returns:
        List các dict — mỗi dict là kết quả compare_models kèm thêm
        key "prompt" chứa prompt gốc.
    """
    results = []
    for prompt in prompts:
        result = compare_models(prompt)
        result["prompt"] = prompt
        results.append(result)
    return results


def format_comparison_table(results: list[dict]) -> str:
    """
    Định dạng kết quả batch_compare thành bảng text dễ đọc.

    Cột: Prompt | GPT-4o Response | Mini Response | GPT-4o Latency | Mini Latency
    Gợi ý: cắt text dài còn 40 ký tự cho dễ nhìn.
    """
    rows = []
    for result in results:
        prompt = result["prompt"]
        gpt4_response = result["gpt4_response"]
        mini_response = result["mini_response"]
        gpt4_latency = result["gpt4_latency"]
        mini_latency = result["mini_latency"]
        rows.append(
            f"{prompt} | {gpt4_response} | {mini_response} | "
            f"{gpt4_latency} | {mini_latency}"
        )
    return "\n".join(rows)


# ---------------------------------------------------------------------------
# Entry point — demo chạy thật (cần OPENAI_API_KEY)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("==================================================================")
    print("      CHẠY CÁC THÍ NGHIỆM VÀ IN Ý CHO 9 CÂU HỎI TRONG MARKDOWN     ")
    print("==================================================================\n")

    # =======================================================================
    # BLOCK 1: API CƠ BẢN
    # =======================================================================
    print("--- [BLOCK 1] API CƠ BẢN ---")

    # Câu 1.1 — Độ nhạy của temperature
    print("\n[Câu 1.1] Chạy call_openai với các mức temperature khác nhau:")
    prompt_hanoi = "Hãy kể cho tôi một sự thật thú vị về Hà Nội."
    temperatures = [0.0, 0.7, 1.2, 1.8]
    for temp in temperatures:
        print(f"\n--- Temperature = {temp} ---")
        try:
            res, lat = call_openai(prompt=prompt_hanoi, temperature=temp, max_tokens=150)
            print(f"Latency: {lat:.2f}s")
            print(f"Response: {res.strip()}")
        except Exception as e:
            print(f"Lỗi khi gọi API: {e}")

    # Câu 1.3 — Đánh đổi chi phí
    print("\n\n[Câu 1.3] Ước tính chi phí (Workload: 20,000 users * 2 calls/day * 500 output tokens):")
    total_tokens = 20000 * 2 * 500  # 20,000,000 tokens
    cost_gpt4o = (total_tokens / 1000) * PRICING_PER_1K_TOKENS["gpt-4o"]["output"]
    cost_mini = (total_tokens / 1000) * PRICING_PER_1K_TOKENS["gpt-4o-mini"]["output"]
    print(f"- Chi phí GPT-4o   : ${cost_gpt4o:.2f} / ngày")
    print(f"- Chi phí GPT-4o-mini: ${cost_mini:.2f} / ngày")


    # =======================================================================
    # BLOCK 2: SYSTEM PROMPT & TOKEN
    # =======================================================================
    print("\n\n--- [BLOCK 2] SYSTEM PROMPT & TOKEN ---")

    # Câu 2.1 — Sức mạnh của persona
    print("\n[Câu 2.1] Chạy chat_with_system_prompt với 2 personas khác nhau:")
    prompt_ml = "Giải thích máy học (machine learning) là gì?"
    persona_poet = "Bạn là một nhà thơ, trả lời mọi thứ bằng hình ảnh ví von, tránh thuật ngữ."
    persona_engineer = "Bạn là kỹ sư phần mềm senior, trả lời chính xác, có ví dụ code khi phù hợp."

    res_poet, _ = chat_with_system_prompt(persona_poet, prompt_ml, max_tokens=200)
    res_eng, _ = chat_with_system_prompt(persona_engineer, prompt_ml, max_tokens=200)

    print(f"\n[Persona Nhà thơ]:\n{res_poet.strip()}")
    print(f"\n[Persona Kỹ sư Senior]:\n{res_eng.strip()}")

    # Câu 2.2 — tiktoken vs đếm từ
    print("\n\n[Câu 2.2] So sánh số token bằng tiktoken vs ước lượng thô (từ / 0.75):")
    text_vn = (
        "Hà Nội là thủ đô của nước Cộng hòa Xã hội Chủ nghĩa Việt Nam, "
        "thành phố lớn nhất Việt Nam về diện tích và cũng là địa phương đứng thứ hai "
        "về dân số. Hà Nội nằm ở trung tâm vùng đồng bằng sông Hồng, là trung tâm "
        "chính trị, văn hóa và là trung tâm kinh tế lớn của đất nước."
    )
    word_count = len(text_vn.split())
    rough_tokens = word_count / 0.75
    exact_tokens = count_tokens(text_vn, model=OPENAI_MODEL)

    print(f"- Đoạn văn thử nghiệm ({word_count} từ):")
    print(f"- Đếm thực tế (tiktoken): {exact_tokens} tokens")
    print(f"- Ước lượng thô (từ / 0.75): {rough_tokens:.1f} tokens")
    diff_pct = abs(exact_tokens - rough_tokens) / exact_tokens * 100
    print(f"- Độ lệch: {diff_pct:.2f}%")


    # =======================================================================
    # BLOCK 4: MINI-PROJECT
    # =======================================================================
    print("\n\n--- [BLOCK 4] MINI-PROJECT ---")

    # Câu 4.1 — Thiết kế persona
    print("\n[Câu 4.1] Demo System Prompt và ràng buộc:")
    persona_demo = (
        "Bạn là trợ lý tư vấn lập trình Python. Hãy giải thích ngắn gọn bản chất vấn đề "
        "và chỉ đưa ra cú pháp/hướng dẫn. (1) Không tạo đoạn mã (code mẫu) hoàn chỉnh. "
        "(2) Trả lời hoàn toàn bằng tiếng Việt."
    )
    print(f"System Prompt được sử dụng:\n\"{persona_demo}\"")

    # Câu 4.2 — Hạn chế & cải thiện + Chạy thử Assistant
    print("\n[Câu 4.2] Chạy mô phỏng trợ lý với max_turns = 1 lượt:")
    # Giả lập input tự động để chạy thử không bị dừng chờ bàn phím
    inputs = iter(["Xin chào, hãy giải thích decorator trong Python ngắn gọn."])
    demo_assistant_stats = run_assistant(
        persona=persona_demo,
        get_input=lambda: next(inputs, "quit"),
        max_turns=1
    )

    print("\n--- Thống kê phiên chat mẫu ---")
    for k, v in demo_assistant_stats.items():
        if k != "history":
            print(f"- {k}: {v}")


    # Demo CLI Part 4
    print("\n\n[Câu 4.3] Chatbot tương tác đa lượt (max_turns = 5) với persona vừa thiết kế:")
    demo_assistant_stats = run_assistant(
        persona=persona_demo,
        max_turns=5,
    )

    print("\n--- Thống kê phiên chat mẫu ---")
    for k, v in demo_assistant_stats.items():
        if k != "history":
            print(f"- {k}: {v}")