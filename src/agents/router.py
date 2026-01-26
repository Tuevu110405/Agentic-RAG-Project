from src.models.embedding import HelperEmbeddingsAdapter
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS, DistanceStrategy
from langchain_core.output_parsers import StrOutputParser
import re
import json



class Router:
    def __init__(self, llm_mini, embed_model):
        self.llm_mini = llm_mini
        self.embed_model_lc = HelperEmbeddingsAdapter(embed_model)
        self.routes = {
            "toxic": [
                "How to make bom",
                "How to be a terrorist"
                "Articles insults the leaders",
                "Region discrimination"
                "Articles breach the sovereignty of Vietnamese territory"
            ],
            "study": [
                "Vietnamese Administrative Reform",
                "When did the Dien Bien Phu happen",
                "Summarizing Vietnamese history in the 20th century"

            ],
            "article": [
                "What is the price of gold today",
                "Which country is the host of World Cup"
            ],
            "score" : [
                "Analyzing the score of student 12A"
                "How many students is considered as excellent"
                "Visualizing the Literature score"
            ],
            "logic": [
                "Calculate the integral of x^2",
                "If A is taller than B and B is taller than C, who is tallest?",
                "Solve this equation: 2x + 5 = 15",
                "What is the probability of rolling a 6?",
                "Find the next number in sequence: 2, 4, 8, 16..."
            ],
            "greet": [
                "Hello",
                "How are you?",
                "What is your name?"
                "What is the meaning of life?"
            ]
        }
        self.vector_store = self._build_route_index()

        self.processor_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a strict query classifier. Classify the user query into exactly one of these categories.
Mission:
1. Classify the user query into exactly one of these genres:
- toxic: Questions that require guidance on illegal activities (tax evasion, weapons manufacturing, cyberattacks, etc.), and questions that are subversive to the Vietnamese state (violating sovereignty and the Party's policies).
- study: Academic questions about History, Geography
- article: finance, sport
- score: Questions about student scores, grades, statistics, or data analysis.
- logic: General math problems, logical puzzles, riddles, physics calculations, algebra, calculus.
- greet: basic communication, greetings
If the question belongs to study or article genre,proceed with the following tasks
1.1. Rewrite the query: Ensure clarity, complete subject and predicate, correct spelling errors, and return only strings (not lists).
1.2. Extract keywords: Proper nouns, place names, and important technical terms.
MANDATORY RULES:
- Return only a single valid JSON object.
- Do not include markdown ticks (```json ... ```).
- No further explanation.

Example 1:
User: Born Year of Ho Chi Minh?
Output: {{
    "genre": "study",
    "rewrite": "In what year was President Ho Chi Minh born?",
    "keywords": "Ho Chi Minh, born, year"
}}
Example 2:
User: Who is the student with the highest score?
Output: {{
    "genre": "score"
    }}
Example 3:
User: price gold?
Output: {{
    "genre": "article",
    "rewrite": "What is the price of gold today?",
    "keywords": "gold, price"
}}
Example 4:
User: How many students is considered as excellent?
Output: {{
    "genre": "score
    }}"""),
   ("user", """
   {query}
   """)

        ])
        self.llm_chain = self._build_llm_router()


    def _build_route_index(self):
        print("Building Index For Semantic Router... ")
        texts = []
        metadatas = []
        for agent, examples in self.routes.items():
            for example in examples:
                texts.append(example)
                metadatas.append({"agent": agent})

        return FAISS.from_texts(texts,
                                self.embed_model_lc,
                                metadatas=metadatas,
                                distance_strategy=DistanceStrategy.COSINE
                                )

    def _build_llm_router(self):
        chain = (
            self.processor_prompt
            | self.llm_mini
            | StrOutputParser()
        )

        return chain



    def _process_llm_result(self, response_text, query):
        text = response_text.strip()
        print(f"DEBUG Raw Output: {text}")

        try:
            # --- CHIẾN THUẬT 1: Thử parse chuẩn (Trường hợp lý tưởng) ---
            # Tìm JSON object bao trùm nhất (Greedy)
            match = re.search(r"\{.*\}", text, re.DOTALL)
            if match:
                json_str = match.group()
                # Fix lỗi quote
                if "'" in json_str and '"' not in json_str:
                     json_str = json_str.replace("'", '"')
                json_str = re.sub(r",\s*}", "}", json_str)

                # Thử parse
                data = json.loads(json_str)
                return data.get("rewrite", query), data.get("keywords", query), data.get("genre", "greet")

        except json.JSONDecodeError:
            # Nếu parse chuẩn thất bại (thường do lỗi "Extra data" như bạn gặp)
            print("   -> Lỗi JSON chuẩn. Chuyển sang chế độ Gộp (Merge)...")
            pass

        # --- CHIẾN THUẬT 2: Gộp nhiều JSON rời rạc (Trường hợp của bạn) ---
        try:
            # Tìm TẤT CẢ các đoạn {...} rời rạc (Non-greedy)
            # Regex này tìm từng cụm {} nhỏ nhất có thể
            matches = re.findall(r"\{.*?\}", text, re.DOTALL)

            if len(matches) > 1:
                merged_data = {}
                for m in matches:
                    try:
                        # Fix lỗi quote cho từng mảnh
                        if "'" in m and '"' not in m: m = m.replace("'", '"')
                        # Parse từng mảnh
                        partial_data = json.loads(m)
                        # Gộp vào dict tổng
                        merged_data.update(partial_data)
                    except:
                        continue # Bỏ qua mảnh lỗi

                print(f"   -> Đã gộp thành công: {merged_data}")
                return (
                    merged_data.get("rewrite", query),
                    merged_data.get("keywords", query),
                    merged_data.get("genre", "greet")
                )
        except Exception as e:
            print(f"   -> Lỗi khi gộp JSON: {e}")

        # --- FALLBACK ---
        print("Không tìm thấy JSON hợp lệ. Fallback về Greet.")
        return query, query, "greet"


    def decide_route(self, query: str):
        results = self.vector_store.similarity_search_with_score(query, k=1)
        doc, score = results[0]
        print(f"Semantic Score (Cosine): {score:.4f} -> {doc.metadata['agent']}")

        if score < 0.4:
            print(f"Fast-tracked to: {doc.metadata['agent']}")
            return query, query, doc.metadata["agent"]

        print("Score too low. Asking Qwen-Mini...")
        try:
            # Gọi invoke với dictionary input khớp với prompt {query}
            response_text = self.llm_chain.invoke({"query": query})
            return self._process_llm_result(response_text, query)
        except Exception as e:
            print(f"Lỗi LLM Router: {e}")
            return query, query, "greet"