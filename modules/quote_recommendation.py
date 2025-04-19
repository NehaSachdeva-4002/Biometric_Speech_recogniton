# modules/quote_recommendation.py
# Module for quote recommendation functionality with LLM enhancement

import json
import random
import logging
import os
from textblob import TextBlob
import torch
from langchain.llms import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from pathlib import Path
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Set up logging
logger = logging.getLogger(__name__)

# Global quotes list
quotes = []

# Initialize the LLM
def initialize_llm():
    """Initialize Hugging Face model for suggestions."""
    try:
        # Load a smaller model suitable for suggestions
        model_id = "distilbert/distilgpt2"  # A lightweight model, replace with your preferred model
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(model_id)
        
        # Create a text generation pipeline
        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_length=100,
            temperature=0.7,
            top_p=0.9
        )
        
        # Create LangChain HF pipeline
        hf_pipe = HuggingFacePipeline(pipeline=pipe)
        
        logger.info("LLM initialized successfully")
        return hf_pipe
    except Exception as e:
        logger.error(f"Failed to initialize LLM: {str(e)}")
        return None

# Global LLM instance
llm = initialize_llm()

def load_quotes():
    """Load quotes from the JSON file."""
    try:
        quotes_path = Path(__file__).parent.parent / 'quotes.json'
        if not quotes_path.exists():
            logger.error(f"Quotes file not found at: {quotes_path}")
            return []
            
        with open(quotes_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            if isinstance(data, dict) and 'quotes' in data:
                quotes = data['quotes']
                logger.info(f"Successfully loaded {len(quotes)} quotes")
                return quotes
            else:
                logger.error("Invalid quotes.json format - missing 'quotes' key")
                return []
    except Exception as e:
        logger.error(f"Error loading quotes: {str(e)}")
        return []

def preprocess_text(text):
    """Preprocess text for comparison."""
    return text.lower().strip()

def get_quote_and_suggestion(user_text):
    """
    Get a relevant quote and suggestion based on user's day description.
    
    Args:
        user_text (str): The user's description of their day
        
    Returns:
        dict: Contains the quote, author, tags, and suggestion
    """
    try:
        # Load quotes
        quotes = load_quotes()
        if not quotes:
            logger.error("No quotes available")
            return {
                'error': 'No quotes available',
                'quote': "Stay positive and keep moving forward.",
                'author': "Unknown",
                'tags': ["inspiration"],
                'suggestion': "Every day is a new opportunity to make progress.",
                'day_type': 'neutral'
            }
            
        # Preprocess texts
        user_text = preprocess_text(user_text)
        
        # Analyze day description
        day_keywords = {
            'good': ['good', 'great', 'wonderful', 'amazing', 'excellent', 'fantastic', 'productive', 'successful'],
            'bad': ['bad', 'terrible', 'awful', 'difficult', 'challenging', 'hard', 'stressful', 'tough'],
            'neutral': ['okay', 'fine', 'normal', 'usual', 'regular', 'average']
        }
        
        # Determine day type
        day_type = 'neutral'
        for type_name, keywords in day_keywords.items():
            if any(keyword in user_text for keyword in keywords):
                day_type = type_name
                break
        
        # Select appropriate tags based on day type
        day_tags = {
            'good': ['success', 'happiness', 'positivity', 'achievement'],
            'bad': ['perseverance', 'courage', 'growth', 'wisdom'],
            'neutral': ['motivation', 'inspiration', 'mindset', 'self-improvement']
        }
        
        # Filter quotes based on day type tags
        relevant_tags = day_tags.get(day_type, [])
        matching_quotes = []
        
        for quote in quotes:
            if isinstance(quote, dict) and 'tags' in quote and any(tag in quote['tags'] for tag in relevant_tags):
                matching_quotes.append(quote)
        
        # Select a quote
        if matching_quotes:
            selected_quote = random.choice(matching_quotes)
        else:
            selected_quote = random.choice(quotes)
        
        # Generate appropriate suggestion based on day type
        suggestions = {
            'good': [
                "Keep up the great work! Your positive energy is inspiring.",
                "Success breeds success - keep this momentum going!",
                "Your achievements today show your potential - keep reaching higher!"
            ],
            'bad': [
                "Every difficult day is an opportunity for growth and learning.",
                "Remember, tough times don't last, but tough people do.",
                "Tomorrow is a new day with new opportunities - keep moving forward."
            ],
            'neutral': [
                "Every day is a chance to make progress, no matter how small.",
                "Consistency is key - keep building your momentum.",
                "Small steps forward are still progress - keep going!"
            ]
        }
        
        suggestion = random.choice(suggestions.get(day_type, suggestions['neutral']))
        
        return {
            'quote': selected_quote.get('quote', "Stay positive and keep moving forward."),
            'author': selected_quote.get('author', "Unknown"),
            'tags': selected_quote.get('tags', ["inspiration"]),
            'suggestion': suggestion,
            'day_type': day_type
        }
        
    except Exception as e:
        logger.error(f"Error in quote recommendation: {str(e)}")
        return {
            'error': 'Failed to get quote recommendation',
            'quote': "Stay positive and keep moving forward.",
            'author': "Unknown",
            'tags': ["inspiration"],
            'suggestion': "Every day is a new opportunity to make progress.",
            'day_type': 'neutral'
        }

def get_sentiment(text):
    """Analyze sentiment of a given text.
    
    Args:
        text (str): The text to analyze
        
    Returns:
        str: "positive", "negative", or "neutral"
    """
    try:
        analysis = TextBlob(text)
        polarity = analysis.sentiment.polarity
        
        if polarity > 0.2:
            return "positive"
        elif polarity < -0.2:
            return "negative"
        else:
            return "neutral"
    except Exception as e:
        logger.error(f"Error in sentiment analysis: {str(e)}")
        return "neutral"

def get_relevant_quote(sentiment):
    """Get a relevant quote based on sentiment.
    
    Args:
        sentiment (str): The sentiment ("positive", "negative", or "neutral")
        
    Returns:
        dict: A quote dictionary with "quote", "author", and "tags" keys
    """
    try:
        sentiment_tags = {
            "positive": ["inspiration", "motivation", "hope", "success", "happiness", "positivity"],
            "negative": ["wisdom", "life", "philosophy", "perseverance", "courage"],
            "neutral": ["humor", "science", "self", "thought", "wisdom"]
        }
        
        relevant_tags = sentiment_tags.get(sentiment, [])
        matching_quotes = []
        
        # Ensure quotes is a list and not empty
        if not isinstance(quotes, list) or not quotes:
            logger.error("Quotes is not a valid list")
            return {"quote": "Stay positive and keep moving forward.", "author": "Unknown", "tags": ["inspiration"]}
            
        for quote in quotes:
            if isinstance(quote, dict) and 'tags' in quote and any(tag in quote['tags'] for tag in relevant_tags):
                matching_quotes.append(quote)
        
        if matching_quotes:
            selected_quote = random.choice(matching_quotes)
            logger.info(f"Selected quote: {selected_quote['quote']}")
            return selected_quote
            
        # If no matching quotes, return a random quote
        selected_quote = random.choice(quotes)
        logger.info(f"Selected random quote: {selected_quote['quote']}")
        return selected_quote
    except Exception as e:
        logger.error(f"Error getting relevant quote: {str(e)}")
        return {"quote": "Stay positive and keep moving forward.", "author": "Unknown", "tags": ["inspiration"]}

def generate_suggestion(sentiment, user_text, quote):
    """Generate a personalized suggestion based on user input and quote.
    
    Args:
        sentiment (str): Detected sentiment
        user_text (str): User's original text
        quote (dict): Selected quote
        
    Returns:
        str: Personalized suggestion or None if generation fails
    """
    if llm is None:
        logger.warning("LLM not available, skipping suggestion generation")
        return None
        
    try:
        # Create a prompt template for the LLM
        template = """
        User sentiment: {sentiment}
        User message: {user_text}
        Quote: "{quote}" - {author}
        
        Based on the user's message and the selected quote, provide a brief, supportive suggestion (1-2 sentences):
        """
        
        prompt = PromptTemplate(
            input_variables=["sentiment", "user_text", "quote", "author"],
            template=template
        )
        
        # Create a chain
        chain = LLMChain(llm=llm, prompt=prompt)
        
        # Run the chain
        response = chain.run(
            sentiment=sentiment,
            user_text=user_text,
            quote=quote["quote"],
            author=quote["author"]
        )
        
        # Clean the response
        suggestion = response.strip()
        logger.info(f"Generated suggestion: {suggestion}")
        
        return suggestion
    except Exception as e:
        logger.error(f"Error generating suggestion: {str(e)}")
        return None

# Load quotes when the module is imported
load_quotes() 