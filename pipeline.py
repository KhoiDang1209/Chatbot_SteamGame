from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
import os
from dotenv import load_dotenv
import re
from google.genai import types
from google import genai
from sentence_transformers import SentenceTransformer

load_dotenv('api.env')


class MongoDBConnection:
    def __init__(self, mongo_access):
        self.mongo_access = os.environ.get("mongodb_access")
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

    @staticmethod
    def extract_metadata_tags(text):
        """
        Extract metadata tags from user query including year, price, and review sentiment.

        Args:
            text (str): User query text

        Returns:
            dict: Dictionary of extracted tags
        """
        tags = {
            "year": None,
            "price": None,
            "review": None
        }
        year_match = re.search(r"\b(19|20)\d{2}\b", text)
        if year_match:
            tags["year"] = int(year_match.group())

        price_match = re.search(r"\$?(\d+)(\.\d{1,2})?", text)
        if price_match:
            tags["price"] = float(price_match.group(1))

        for sentiment in ["Positive", "Mixed", "Negative"]:
            if sentiment.lower() in text.lower():
                tags["review"] = sentiment
                break

        return tags

    def vector_search_description(self, limit=100):
        """
        Perform a vector search and filter results using extracted metadata.

        Args:
            limit (int): Maximum number of results to return

        Returns:
            list: Filtered results based on vector search and metadata
        """
        # Step 1: Extract metadata tags from user query
        tags = DataHandler.extract_metadata_tags(self.user_query)

        # Step 2: Generate embedding for the query
        query_embedding = self.embedding_model.get_embedding(self.user_query)
        if not query_embedding:
            return "Invalid query or embedding generation failed."

        # Step 3: Run vector search
        pipeline = [
            {
                "$vectorSearch": {
                    "index": "des_embed",
                    "queryVector": query_embedding,
                    "path": "embedding_description",
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

        results = list(self.collection.aggregate(pipeline))

        filtered_results = []
        for game in results:
            # Check year
            if tags["year"] is not None:
                if not game.get("release_date") or game["release_date"].year != tags["year"]:
                    continue

            # Check price
            if tags["price"] is not None:
                if not game.get("price"):
                    continue
                try:
                    game_price = float(str(game["price"]).replace("$", "").strip())
                    if game_price > tags["price"]:
                        continue
                except ValueError:
                    continue  # Skip games with malformed price

            # Check review
            if tags["review"]:
                if not game.get("all_reviews") or tags["review"].lower() not in game["all_reviews"].lower():
                    continue

            # If all filters pass
            filtered_results.append(game)

        return filtered_results[:5] if filtered_results else results[:5]

    def vector_search_name(self, limit=1):
        """
        Perform a vector search in the MongoDB collection based on the game name.

        Args:
            limit (int): Maximum number of results to return

        Returns:
            list: Results of the vector search
        """
        # Generate embedding for the user query
        query_embedding = self.embedding_model.get_embedding(self.user_query)

        if not query_embedding:
            return "Invalid query or embedding generation failed."

        # Define the vector search pipeline
        pipeline = [
            {
                "$vectorSearch": {
                    "index": "name_embed",
                    "queryVector": query_embedding,
                    "path": "embedding_name",
                    "numCandidates": 400,
                    "limit": limit,
                }
            },
            {
                "$unset": "embedding"
            },
            {
                "$project": {
                    "_id": 0,
                    "name": 1,  # Game name
                    "description": 1,  # Game description
                    "all_reviews": 1,  # Game review
                    "release_date": 1,  # Release date
                    "developer": 1,  # Developer name
                    "publisher": 1,  # Publisher name
                    "price": 1,  # Game price
                    "score": {
                        "$meta": "vectorSearchScore"
                    }
                }
            }
        ]

        # Execute the search
        results = list(self.collection.aggregate(pipeline))
        return results


class ModelResponse:
    def __init__(self, gemini_api_key):
        self.client = genai.Client(api_key=gemini_api_key)

    def generate_response(self, user_query, retrieved_games):
        """
        Generate a recommendation response using Gemini Flash API.

        Args:
            user_query (str): The original user query.
            retrieved_games (list): List of game metadata dicts.

        Returns:
            str: The generated recommendation text.
        """
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
        """
        Process Gemini Flash function call and generate final response.

        Args:
            user_query (str): The original user query.
            collection: The MongoDB collection to search.
            embedding_model: The embedding model to use for vector search.

        Returns:
            str: The generated recommendation text or error message.
        """
        function_declarations = [
            {
                "name": "vector_search_name",
                "description": "Search games by their title or name similarity",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "User input that includes a game title or specific name"
                        }
                    },
                    "required": ["query"]
                }
            },
            {
                "name": "vector_search_description",
                "description": "Search games based on genre, style, or gameplay description",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "User input that includes preferences or genres"
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

            # Create a data handler with the query from function call
            data_handler = DataHandler(
                function_call.args["query"],
                collection,
                embedding_model
            )

            if function_call.name == "vector_search_name":
                results = data_handler.vector_search_name()
            elif function_call.name == "vector_search_description":
                results = data_handler.vector_search_description()
            else:
                return "❌ Unknown function"

            return self.generate_response(user_query, results)
        else:
            print("⚠️ No function call found.")
            if hasattr(candidate.content.parts[0], "text"):
                return candidate.content.parts[0].text
            return "No appropriate response found."