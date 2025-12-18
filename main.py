import os
import logging
import asyncio
from flask import Flask, request, jsonify
from threading import Thread

# Настройка логирования
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Flask приложение
app = Flask(__name__)

# Конфигурация
TOKEN = os.getenv('TOKEN', '8255764534:AAH6gMVaBXsctXqRUM5VujJM-O-cWKuiuRM')
WEBHOOK_URL = f"https://multiverse-rp-bot.onrender.com/webhook"

# Глобальная переменная для приложения бота
application = None

async def setup_bot():
    """Асинхронная настройка бота"""
    global application
    
    try:
        from telegram.ext import Application as TelegramApp, CommandHandler, MessageHandler, filters
        
        # Создаем приложение БЕЗ updater
        application = TelegramApp.builder().token(TOKEN).build()
        
        # Импортируем обработчики
        from telegram import Update
        from telegram.ext import ContextTypes
        
        async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
            await update.message.reply_text("✅ Бот работает с вебхуками!")
            
        async def help_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
            await update.message.reply_text("Список команд:\n/start - Проверка\n/help - Помощь")
            
        # Добавляем обработчики
        application.add_handler(CommandHandler("start", start))
        application.add_handler(CommandHandler("help", help_cmd))
        
        # Инициализируем
        await application.initialize()
        
        # Устанавливаем вебхук
        await application.bot.set_webhook(
            url=WEBHOOK_URL,
            drop_pending_updates=True,
            allowed_updates=["message", "callback_query"]
        )
        logger.info(f"✅ Вебхук установлен: {WEBHOOK_URL}")
        
        return application
        
    except Exception as e:
        logger.error(f"Ошибка настройки бота: {e}")
        raise

def run_async_setup():
    """Запуск асинхронной настройки"""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(setup_bot())
    finally:
        loop.close()

@app.route('/webhook', methods=['POST'])
def webhook():
    """Обработчик вебхуков"""
    try:
        if request.method == 'POST' and application:
            # Получаем обновление
            update_data = request.get_json()
            
            if update_data:
                # Создаем новую event loop для обработки
                async def process():
                    from telegram import Update
                    update = Update.de_json(update_data, application.bot)
                    await application.process_update(update)
                
                # Запускаем в отдельном потоке
                thread = Thread(target=lambda: asyncio.run(process()))
                thread.daemon = True
                thread.start()
                
            return jsonify({'status': 'ok'}), 200
            
    except Exception as e:
        logger.error(f"Ошибка в вебхуке: {e}")
        return jsonify({'status': 'ok'}), 200  # Всегда возвращаем 200

@app.route('/')
def home():
    return jsonify({'status': 'bot_is_running', 'mode': 'webhook'})

@app.route('/setwebhook')
def set_webhook():
    """Установка вебхука вручную"""
    try:
        run_async_setup()
        return jsonify({'status': 'webhook_set', 'url': WEBHOOK_URL}), 200
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

def main():
    """Основная функция"""
    logger.info("🚀 Запуск бота на Flask...")
    
    # Настраиваем бота в отдельном потоке
    setup_thread = Thread(target=run_async_setup)
    setup_thread.daemon = True
    setup_thread.start()
    
    # Запускаем Flask
    port = int(os.getenv('PORT', 8080))
    app.run(host='0.0.0.0', port=port, debug=False, threaded=True)

if __name__ == "__main__":
    main()
