import gradio as gr
def ui(app_instance):
    def generate_response(message, history):
        try:
            response = app_instance.flow(message)
            return response
        except Exception as e:
            return f"Error {e}"

    demo = gr.ChatInterface(
        fn = generate_response,
        title = "Agentic RAG System",
        description = """Brilliant Agentic System Answering question in different domain
        - **Study**: History, Geography
        - **Article**: Sport, Gold Price
        - **Logic**: Math, Logical puzzle(coding support)
        - **Score**: Student score analysis
        """,
        theme= "soft",
        
        examples=[
            "Chiến thắng Điện Biên Phủ năm nào?",          # Study
            "Giá vàng hôm nay thế nào?",                  # Article
            "Tính tổng của 15 và 25?",                    # Logic
            "Học sinh nào có điểm cao nhất?",             # Score
            "Hello bot",                                  # Greet
            "Giải phương trình 2x + 5 = 15"               # Logic
        ],
        cache_examples=False,
        
    )
    return demo