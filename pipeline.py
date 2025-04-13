from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
from dotenv import load_dotenv
from google.genai import types
from google import genai
from datetime import datetime



load_dotenv('api.env')


class MongoDBConnection:
    def __init__(self, mongo_access):
        self.mongo_access = mongo_access
        # Create a new client and connect to the server
        self.client = MongoClient(mongo_access, server_api=ServerApi('1'))
        self.db = self.client["Steam_Game"]
        self.collection = self.db["Steam_Embedding"]

        # Test connection
        try:
            self.client.admin.command('ping')
            print("Pinged your deployment. You successfully connected to MongoDB!")
        except Exception as e:
            print(e)


class EmbeddingModelSentence:
    def __init__(self, embedding_model):
        self.embedding_model = embedding_model

    def get_embedding(self, text):
        if not isinstance(text, str) or not text.strip():
            print("Skipping invalid text")
            return []
        return self.embedding_model.encode(text).tolist()


class DataHandler:
    def __init__(self, user_query, collection, embedding_model):
        self.user_query = user_query
        self.collection = collection
        self.embedding_model = embedding_model


    def smart_vector_search(self,
            query: str,
            collection,
            year_range: list[int] = None,
            price_limit: float = None,
            review_sentiment: str = None,
            developer: str = None,
            publisher: str = None,
            limit: int = 100
    ):
        # Step 1: Embed the query
        query_embedding = self.embedding_model.get_embedding(query)
        if not query_embedding:
            return []

        # Step 2: Run vector search in MongoDB
        pipeline = [
            {
                "$vectorSearch": {
                    "index": "default",
                    "queryVector": query_embedding,
                    "path": "embedding",
                    "numCandidates": 400,
                    "limit": limit,
                }
            },
            {"$unset": "embedding"},
            {
                "$project": {
                    "_id": 0,
                    "name": 1,
                    "description": 1,
                    "all_reviews": 1,
                    "release_date": 1,
                    "developer": 1,
                    "publisher": 1,
                    "price": 1,
                    "score": {"$meta": "vectorSearchScore"},
                }
            }
        ]

        results = list(collection.aggregate(pipeline))

        # Step 3: Post-filter
        filtered_results = []
        for game in results:
            # Release year filtering
            if year_range and isinstance(game.get("release_date"), datetime):
                year = game["release_date"].year
                if not (year_range[0] <= year <= year_range[1]):
                    continue

            # Price filtering
            if price_limit is not None:
                price_str = str(game.get("price", "")).replace("$", "").strip()
                try:
                    game_price = float(price_str)
                    if game_price > price_limit:
                        continue
                except ValueError:
                    continue

            # Review sentiment filtering
            if review_sentiment and review_sentiment.lower() not in game.get("all_reviews", "").lower():
                continue

            # Developer match
            if developer and developer.lower() not in game.get("developer", "").lower():
                continue

            # Publisher match
            if publisher and publisher.lower() not in game.get("publisher", "").lower():
                continue

            filtered_results.append(game)

        return filtered_results[:3] if filtered_results else results[:3]



class ModelResponse:
    def __init__(self, gemini_api_key):
        self.client = genai.Client(api_key=gemini_api_key)

    def generate_response(self, user_query, retrieved_games):
        if not retrieved_games or isinstance(retrieved_games, str):
            return "Sorry, I couldn't find any games matching your query."

        context = "\n".join([
            f"{g['name']}: {g['description']}: {g['all_reviews']}: {g['release_date']}: {g['publisher']}: {g['price']}"
            for g in retrieved_games if isinstance(g, dict)
        ])

        prompt = f"""
        You are a game recommendation agent. Your task is to provide engaging and convincing recommendations to users based on their queries and the following retrieved game information.

        User Query: {user_query}

        Relevant Games:
        {context}

        Provide a detailed recommendation that:
        - Highlights the most appealing aspects of the games.
        - Connects the games to the user's query.
        - Uses persuasive language and your own knowledge to captivate the user.
        - Includes relevant information such as gameplay, reviews, release date, publisher and price.
        - Make the user want to play the game.
        """

        response = self.client.models.generate_content(
            model="gemini-2.0-flash",
            contents=prompt
        )

        return response.text

    def process_response(self, user_query, collection, embedding_model):

        function_declarations = [
            {
                "name": "vector_search_filtered",
                "description": "Search games based on description and apply filters like year range, price, review sentiment, developer, or publisher.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "User's game preferences (e.g., football tactical game)"
                        },
                        "year_range": {
                            "type": "array",
                            "items": {"type": "integer"},
                            "description": "Optional release year range (e.g., [2020, 2024])"
                        },
                        "price_limit": {
                            "type": "number",
                            "description": "Maximum price (e.g., 10 for under $10)"
                        },
                        "review_sentiment": {
                            "type": "string",
                            "enum": ["Positive", "Mixed", "Negative"],
                            "description": "Preferred review sentiment"
                        },
                        "developer": {
                            "type": "string",
                            "description": "Specific developer or studio name (e.g., 'Ubisoft')"
                        },
                        "publisher": {
                            "type": "string",
                            "description": "Specific publisher name (e.g., 'SEGA')"
                        }
                    },
                    "required": ["query"]
                }
            },
            {
                "name": "chit_chat",
                "description": "Chit chat message of users",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "User input message that is common chit-chat"
                        }
                    },
                    "required": ["query"]
                }
            },
            {
                "name": "end_chat",
                "description": "End chat session",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "End the chat session and further information if needed"
                        }
                    },
                    "required": ["query"]
                }
            }

        ]
        tools = types.Tool(function_declarations=function_declarations)
        config = types.GenerateContentConfig(tools=[tools])

        response = self.client.models.generate_content(
            model="gemini-2.0-flash",
            contents=user_query,
            config=config,
        )
        candidate = response.candidates[0]

        if candidate.content.parts and candidate.content.parts[0].function_call:
            function_call = candidate.content.parts[0].function_call
            print(f"🔧 Function to call: {function_call.name}")
            print(f"📥 Arguments: {function_call.args}")

            if function_call.name == "vector_search_filtered":
                args = function_call.args
                data_handler= DataHandler(user_query, collection, embedding_model)
                results = data_handler.smart_vector_search(
                    query=user_query,
                    collection=collection,
                    year_range=args.get("year_range"),
                    price_limit=args.get("price_limit"),
                    review_sentiment=args.get("review_sentiment"),
                    developer=args.get("developer"),
                    publisher=args.get("publisher")
                )
            elif function_call.name == "chit-chat":
                chit_chat_response = self.client.models.generate_content(
                    model="gemini-2.0-flash",
                    contents=function_call.args["query"],
                )
                return chit_chat_response.text
            else:
                end_response = self.client.models.generate_content(
                    model="gemini-2.0-flash",
                    contents=function_call.args["query"],
                )
                return end_response.text

            # Use local model to generate the response
            return ModelResponse.generate_response(self,user_query, results)