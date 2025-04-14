# reflection.py
class Reflection:
    def __init__(self, llm):
        self.llm = llm

    def concat_and_format_texts(self, chat_history):
        lines = []
        for msg in chat_history:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            lines.append(f"{role}: {content}\n")
        return ''.join(lines)

    # reflection.py
    def get_standalone_query(self, chat_history,new_prompt):
        history_string = self.concat_and_format_texts(chat_history)
        prompt = f"""
        You are a system designed to reformulate user messages into standalone game queries. Be friendly and concise. Your task is to determine if the latest user message depends on prior context.

        Given the following chat history and the latest user message, determine if the latest message depends on prior context. If it does, rewrite it into a standalone game query. If it is already standalone or the chat history is empty, return the original message without any additional text.

        Chat History:
        {history_string}
        New prompt:
        {new_prompt}
        Return only the rewritten message or the original message. Do not include any explanations, recommendations, or additional text.
        """

        response = self.llm.models.generate_content(
            model="gemini-2.0-flash",
            contents=prompt
        )
        print(prompt)

        return response.text.strip()
