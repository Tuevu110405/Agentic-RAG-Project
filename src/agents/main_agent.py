from src.agents.router import Router
from langchain_experimental.utilities import PythonREPL
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
import re
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from src.config import config



class Main:
    def __init__(self, llm_mini, llm_large, vector_db, embed_model, llm_cloud):
        self.llm_mini = llm_mini
        self.query_processor = Router(llm_mini, embed_model)
        self.llm_large = llm_large
        self.vector_db = vector_db
        self.python_repl = PythonREPL()
        self.llm_cloud = llm_cloud
        self.answer_prompt_study = ChatPromptTemplate.from_messages([
            ("system", """You are a helpful AI assistant; your task is to answer question from user related to history and geography
SAFETY AND HONESTY RULES:
1. Only use the information in the "Context" section below to answer.
2. If you cannot find the information in the context, use your own knowledge, but prioritize the context.


OUTPUT FORMAT:
- Briefly step-by-step reasoning.

"""),
            ("user", """Context:
{context}

Query:
{question}
""")
        ])

        self.answer_prompt_article = ChatPromptTemplate.from_messages([
            ("system", """You are a helpful AI assistant; your task is to answer question from user related to sport and finance
SAFETY AND HONESTY RULES:
1. Only use the information in the "Context" section below to answer.
2. If you cannot find the information in the context, use your own knowledge, but prioritize the context.

OUTPUT FORMAT:
- Briefly step-by-step reasoning.

"""),
            ("user", """Context:
{context}

Query:
{question}
""")
        ])
        self.answer_prompt_greet = ChatPromptTemplate.from_messages([
            ("system", """You are a helpful AI assistant; your task is to greet users and introduce you as an assistant of answer question from user related to study, article, math, student score
            RULES: Remember to answer with positive thought
"""
)
        ])

    #rag direction methods
    #format docs resulted from retrieving(rag direction)
    def format_docs(self, docs):
        return "\n\n".join(f"{doc.page_content}" for doc in docs)

    #math direction methods
    #extract code generated from llm
    def extract_code(self,text):
        # Bước 1: Làm sạch chuỗi đầu vào (đôi khi Gemini trả về ```json ở ngoài cùng)
        clean_text = text.strip()
        if clean_text.startswith("```json"):
            clean_text = clean_text.replace("```json", "").replace("```", "")

        # Bước 2: Thử Parse JSON
        # Vì output của bạn là {"answer": "..."} nên ta cần lấy nội dung bên trong key "answer"
        try:
            data = json.loads(clean_text)
            if isinstance(data, dict):
                # Ưu tiên lấy từ key 'answer', nếu không có thì lấy chuỗi gốc
                text = data.get("answer", text)
            elif isinstance(data, str):
                # Trường hợp parse ra trực tiếp string
                text = data
        except json.JSONDecodeError:
            # Nếu không phải JSON hợp lệ thì giữ nguyên text cũ để regex xử lý
            pass

        # Bước 3: Dùng Regex linh hoạt hơn
        # - (?:python|py)? : Chấp nhận ```python, ```py hoặc chỉ ```
        # - \s+ : Chấp nhận mọi khoảng trắng (xuống dòng \n, dấu cách) sau thẻ mở
        # - (.*?) : Nội dung code
        # - \s* : Bỏ qua khoảng trắng thừa ở cuối trước khi đóng ```
        pattern = r"```(?:python|py)?\s+(.*?)\s+```"

        match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1).strip()

        # Fallback: Nếu không tìm thấy pattern ```...``` nhưng text đã được extract từ JSON
        # và có vẻ giống code (có import, print...), ta trả về text đó luôn.
        if "import " in text or "print(" in text:
            return text.strip()

        return None


    #study direction
    def get_answer_study(self, question_text, rewrite, keywords, top_k=5, weights=[0.7, 0.3]):
        if isinstance(rewrite, list):
            rewrite = " ".join(rewrite)

        print(f"DEBUG-Rewrite: {rewrite}")
        print(f"DEBUG-Keywords: {keywords}")

        docs = self.vector_db.hybrid_search(#query = rewritten_query,

                                            query_dense = rewrite,
                                            query_sparse = keywords,
                                            top_k=top_k,
                                            weights=weights,
                                            collection_name = 'study')
        context_text = self.format_docs(docs)

        chain = (
            {
                "context" : lambda x: context_text,
                "question": lambda x : question_text
            }
            | self.answer_prompt_study
            | self.llm_large
        )

        return chain.invoke(question_text)

    def get_answer_article(self, question_text, rewrite, keywords, top_k=5, weights=[0.7, 0.3]):
        if isinstance(rewrite, list):
            rewrite = " ".join(rewrite)

        print(f"DEBUG-Rewrite: {rewrite}")
        print(f"DEBUG-Keywords: {keywords}")

        docs = self.vector_db.hybrid_search(#query = rewritten_query,

                                            query_dense = rewrite,
                                            query_sparse = keywords,
                                            top_k=top_k,
                                            weights=weights,
                                            collection_name = 'article')
        context_text = self.format_docs(docs)

        chain = (
            {
                "context" : lambda x: context_text,
                "question": lambda x : question_text
            }
            | self.answer_prompt_article
            | self.llm_large
        )

        return chain.invoke(question_text)

    def get_answer_toxic(self, question_text):
        chain = (
            {
                "question": lambda x : question_text

            }
            | self.answer_prompt_toxic
            | self.llm_large
        )
        return chain.invoke(question_text)

    def get_answer_greet(self, question_text):
        chain = (
            {
                "question": lambda x : question_text,
            }
            | self.answer_prompt_greet
            | self.llm_mini
        )
        return chain.invoke(question_text)

    #math direction of TA
    def _clean_code(self, code):
        """Tự động thêm import hoặc sửa lỗi cú pháp cơ bản"""
        if "import math" not in code:
            code = "import math\nimport json\n" + code
        return code

    # --- MAIN METHOD: NÂNG CẤP ---
    def get_answer_math_logic(self, question_text, max_retries=3):
        # 1. Khởi tạo lịch sử hội thoại (Memory ngắn hạn)
        # Lưu ý: Chúng ta dùng list Message để duy trì ngữ cảnh cho việc sửa lỗi
        messages = [
            SystemMessage(content="""You are a helpful AI assistant; your task is to answer questions by writing Python code that accurately solves the user's question.

1. Write Python code to SOLVE the user's logic puzzle, math problem, or data request.
2. The code MUST print a friendly, natural language response explaining the result.
   - Bad: `print(x)`
   - Bad: `print("Answer: A")`
   - Good: `print(f"I calculated it, and the total is {total}. This is because...")`

Rules for Code:
1. The name of variable: Avoid using Python keywords (`lambda`, `class`, `return`, `min`, `max`, `sum`...). Should use `var_x`, `total_v`,...
Example:
Example A: The "Who is Tallest?" Logic Puzzle
User: "An is taller than Binh. Chi is taller than An. Who is the tallest?"

```python
# 1. Define relative heights (using an arbitrary base)
# Let Binh = 100 units
heights = {{}}
heights['Binh'] = 100
heights['An'] = heights['Binh'] + 10  # An is taller than Binh
heights['Chi'] = heights['An'] + 10    # Chi is taller than An

# 2. Sort to find the tallest
sorted_people = sorted(heights.items(), key=lambda x: x[1], reverse=True)
tallest_name = sorted_people[0][0]

# 3. Print a conversational explanation
print(f"Based on your description, **{{tallest_name}}** is the tallest.")
print("Here is the order from tallest to shortest:")
for name, height in sorted_people:
    print(f"- {{name}}")

```
Example B: The "Chicken and Rabbit" Problem (Algebra)
User: "A farm has 35 heads and 94 legs. How many chickens and rabbits?"
```python
from sympy import symbols, Eq, solve

# 1. Setup Symbols
c, r = symbols('c r') # c=chickens, r=rabbits

# 2. Equations
# Heads: c + r = 35
# Legs:  2c + 4r = 94
eq1 = Eq(c + r, 35)
eq2 = Eq(2*c + 4*r, 94)

# 3. Solve
sol = solve((eq1, eq2), (c, r))
chickens = sol[c]
rabbits = sol[r]

# 4. Conversational Output
print(f"I solved the system of equations and found the answer:")
print(f"- **{{rabbits}} Rabbits**")
print(f"- **{{chickens}} Chickens**")
print(f"Check: {{chickens}} + {{rabbits}} = 35 heads, and 2*{{chickens}} + 4*{{rabbits}} = 94 legs.")
```
Example C: General Math / Probability
User: "What is the probability of rolling a sum of 7 with two dice?"
```python
# 1. Calculate all outcomes
outcomes = [(d1, d2) for d1 in range(1, 7) for d2 in range(1, 7)]
total_combinations = len(outcomes)

# 2. Find winners (Sum = 7)
winners = [pair for pair in outcomes if sum(pair) == 7]
count = len(winners)

# 3. Calculate Stats
prob = count / total_combinations
percent = prob * 100

# 4. Conversational Output
print(f"There are {{total_combinations}} possible combinations for two dice.")
print(f"A sum of 7 appears {{count}} times: {{winners}}.")
print(f"So, the probability is **{{count}}/{{total_combinations}}** (approx **{{percent:.1f}}%**).")
```


"""

),
            HumanMessage(content=f"Câu hỏi: {question_text}")
        ]


        # 2. Vòng lặp Suy luận & Sửa lỗi (ReAct Loop)
        # 2. Vòng lặp Suy luận & Sửa lỗi (ReAct Loop)
        for attempt in range(max_retries):
            print(f"   [Logic] Attempt {attempt + 1}/{max_retries}...")

            # Bước A: Gọi LLM để sinh code
            try:
                ai_msg = self.llm_cloud.invoke(messages)
                print(f"   [Logic] LLM Response: {ai_msg.content}")
            except Exception as e:
                print(f"   [Logic] LLM Error: {e}")
                return "I apologize, but I encountered an error while processing your request."

            content = ai_msg.content if hasattr(ai_msg, 'content') else str(ai_msg)
            messages.append(ai_msg) # Lưu message của AI vào lịch sử

            # Bước B: Trích xuất Code
            code_block = self.extract_code(content)

            if not code_block:
                # Nếu không có code, có thể AI đã trả lời trực tiếp bằng lời
                # Nếu là lượt cuối cùng, trả về luôn nội dung đó
                if attempt == max_retries - 1:
                    return content

                # Nếu chưa phải lượt cuối, nhắc AI viết code (vì ta đang ở trong Logic Agent)
                print("The code can not be extracted")
                messages.append(HumanMessage(content="You didn't provide any Python code. Please write the Python code to calculate the answer."))
                continue

            # Bước C: Thực thi Code
            code_block = self._clean_code(code_block)
            print(f"   [Logic] Executing Code...")

            try:
                # Chạy code và lấy output (print)
                exec_result = self.python_repl.run(code_block)
                exec_result = str(exec_result).strip()
                print(f"   [Logic] Output: {exec_result}")

                # --- THAY ĐỔI QUAN TRỌNG: XỬ LÝ KẾT QUẢ ---

                # 1. Nếu code chạy nhưng không in ra gì
                if not exec_result:
                    feedback = "The code executed successfully but printed nothing. Please rewrite the code to PRINT the final result."
                    messages.append(HumanMessage(content=feedback))
                    continue

                # 2. Nếu có kết quả: Đưa kết quả lại cho LLM để sinh câu trả lời tự nhiên
                # Đây là bước "Reasoning based on Tool Output"
                interpretation_prompt = (
                    f"Code executed successfully. Output:\n{exec_result}\n\n"
                    "Based on this output, please provide a clear, natural language answer to the user's question."
                )

                # Gọi LLM lần cuối để diễn giải kết quả
                messages.append(HumanMessage(content=interpretation_prompt))
                final_response_msg = self.llm_cloud.invoke(messages)

                return final_response_msg.content # Trả về câu trả lời tự nhiên

            except Exception as e:
                # Bước E: Xử lý lỗi (Self-Correction)
                error_msg = str(e)
                print(f"   [Logic] Runtime Error: {error_msg}")

                # Gửi lỗi lại cho LLM để nó tự sửa code
                fix_prompt = f"The code encountered an error: {error_msg}. Please rewrite the Python code to fix this."
                messages.append(HumanMessage(content=fix_prompt))
                continue

        # 3. Fallback (Nếu hết lượt mà vẫn lỗi)
        print("   [Logic] Retries exhausted.")
        return "I tried to run the calculation multiple times but encountered errors. Please check the logic."

    def get_answer_score(self, question_text, csv_path = config.csv_path, max_tries=3):

        messages = [
        SystemMessage(content=f"""You are a Python Expert and Data Analyst.


RESOURCES:
- You have a CSV file at: '{csv_path}'
- Columns: "name", "score"

TASK:
1. Write Python code to load the CSV using pandas.
2. Calculate the answer based on the user's query.
3. The code MUST print the final answer in a friendly, natural language format.
   - ❌ Bad: `print(5)`
   - ❌ Bad: `print("Answer: A")`
   - ✅ Good: `print(f"The average Math score is {{avg_score:.2f}}.")`
   - ✅ Good: `print(f"The student with the highest score is {{name}} with {{score}} points.")`

TIPS:
- Use `pd.read_csv('{csv_path}')` to load data.
- Handle potential empty results gracefully.
- If the user asks for a list, print it nicely (e.g., bullet points).

Student Evaluation:
- Student who having score bigger than 90 is excellent student.
- Student who having score bigger than 80 is good student.
- Student who having score bigger than 70 is average student.
- Student who having score smaller than 70 is bad student.

Return ONLY the Python code block.


"""

),
            HumanMessage(content=f"Câu hỏi: {question_text}")
        ]


        # 2. Vòng lặp Suy luận & Sửa lỗi (ReAct Loop)
        # 2. Vòng lặp Suy luận & Sửa lỗi (ReAct Loop)
        for attempt in range(max_tries):
            print(f"   [Score Agent] Attempt {attempt+1}/{max_tries}...")

            # 1. Get Code from LLM
            try:
                ai_msg = self.llm_cloud.invoke(messages)
                messages.append(ai_msg)
            except Exception as e:
                return f"System Error: {e}"
            content = ai_msg.content if hasattr(ai_msg, 'content') else str(ai_msg)
            # 2. Extract Code
            code = self.extract_code(content)
            if not code:
                # If LLM replied with text only, return it (it might be a simple refusal or answer)
                return content

            # 3. Execute Code
            try:
                # Add import if missing
                if "import pandas" not in code:
                    code = "import pandas as pd\n" + code

                output = self.python_repl.run(code)
                print(f"   [Output]: {output}")

                # Check if output is valid
                if output.strip():
                    return output.strip() # Return the print output directly
                else:
                    # Feedback loop: Code ran but printed nothing
                    messages.append(HumanMessage(content="The code ran successfully but printed nothing. Please modify the code to PRINT the final answer string."))

            except Exception as e:
                print(f"   [Error]: {e}")
                # Feedback loop: Code failed
                messages.append(HumanMessage(content=f"Runtime Error: {e}. Please fix the python code."))

        return "I tried to analyze the data but encountered technical errors."


    def flow(self, q_text):
        """
        Hàm chính nhận 1 dictionary từ file JSON và trả về câu trả lời.

        """
        final_answer = ""



        # 1. Format choices thành string A. ... B. ...


        # 2. Router & Processing (Gọi 1 lần duy nhất)

        rewrite, keywords, genre = self.query_processor.decide_route(q_text)
        print(f"ROUTER DECISION: {genre}")

        # 3. Routing


        try:
            if genre == "logic":
                final_answer = self.get_answer_math_logic(q_text)

            elif genre == "study":
                final_answer = self.get_answer_study(q_text, rewrite, keywords)
            elif genre == "article":
                final_answer = self.get_answer_article(q_text, rewrite, keywords)
            elif genre == "toxic":
                final_answer = self.get_answer_toxic(q_text)
            elif genre == "score":
                final_answer = self.get_answer_score(q_text)


            else: # default as greet
                final_answer = self.get_answer_greet(q_text)

        except Exception as e:
            print(f"CRITICAL ERROR in flow: {e}")

        return final_answer