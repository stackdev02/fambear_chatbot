import os
import requests
import logging
import json
from collections import deque
from dotenv import load_dotenv
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, CallbackContext

# Load environment variables
load_dotenv()
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
AWS_LAMBDA_URL = os.getenv("AWS_LAMBDA_URL")

# Set up logging
logging.basicConfig(level=logging.INFO)

# Store last 20 messages per user
user_histories = {}

async def start(update: Update, context: CallbackContext):
    """Handle /start command."""
    await update.message.reply_text("Hello! Send me a message, then I can help you to find right candidates for your family.")

async def handle_message(update: Update, context: CallbackContext):
    """Handle user messages and forward to AWS Lambda."""
    chat_id = update.message.chat_id
    user_message = update.message.text

    # Maintain conversation history (store only the last 20 messages)
    if chat_id not in user_histories:
        user_histories[chat_id] = deque(maxlen=20)

    user_histories[chat_id].append({"role": "user", "content": user_message})

    # Show typing indicator while processing
    await context.bot.send_chat_action(chat_id=chat_id, action="typing")

    # Send request to AWS Lambda with chat history
    payload = {"chat_id": chat_id, "history": list(user_histories[chat_id])}

    try:
        response = requests.post(AWS_LAMBDA_URL, json=payload, timeout=59)


        # Check if response is successful
        if response.status_code == 200:
            data = response.json()
            body = json.loads(data.get("body"))
            chatgpt_response = body.get("reply")
            p_time = body.get("p_time")

            if chatgpt_response:
                await update.message.reply_text(f"{chatgpt_response} ({p_time}s)")

                # Add ChatGPT response to history
                user_histories[chat_id].append({"role": "assistant", "content": chatgpt_response})
            else:
                logging.error(f"Invalid response format: {data}")
                await update.message.reply_text("Sorry, I received an unexpected response.")
        else:
            error_message = response.json().get("error")
            if error_message:
                logging.error(f"AWS Lambda Error: {error_message}")
                await update.message.reply_text(f"AWS Lambda Error: {error_message}")
            else:
                logging.error(f"Lambda API Error: {response.status_code} - {response.text}")
                await update.message.reply_text("Sorry, something went wrong with the server.")

    except requests.exceptions.RequestException as e:
        logging.error(f"Request to AWS Lambda failed: {e}")
        await update.message.reply_text("Network error. Please try again later.")

def main():
    """Start the bot."""
    app = Application.builder().token(TELEGRAM_BOT_TOKEN).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    # Run the bot
    app.run_polling()

if __name__ == "__main__":
    main()
