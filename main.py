import logging
import json
import os
import re
import string
import datetime
import asyncio
import random
from functools import wraps
from typing import Literal, Optional, List, Dict, Union

from dotenv import load_dotenv
load_dotenv()

from sqlalchemy import create_engine, Column, Integer, String, Text, Boolean, DateTime, Date, ForeignKey, func, or_, and_, text, BigInteger
from sqlalchemy.orm import sessionmaker, relationship, declarative_base
from sqlalchemy.types import TypeDecorator, CHAR
from sqlalchemy import TypeDecorator, String as SQLA_String
import uuid

from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup, User as TGUser, CallbackQuery, Message
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    CallbackQueryHandler,
    filters,
    ContextTypes,
    JobQueue,
    ConversationHandler,
)
from telegram.error import TelegramError, TimedOut

TOKEN = os.getenv('TOKEN', '8255764534:AAH6gMVaBXsctXqRUM5VujJM-O-cWKuiuRM')
DEVELOPER_IDS = [6283690984]
INITIAL_ANKETNIK_USER_IDS = [6283690984]

ANKET_CHANNEL_ID = -1003394079022
DEVELOPER_CHAT_ID = 6283690984
ANKETNIK_CHAT_ID = -1003394079022

LOGGING_CHAT_IDS = [
    -1003431402721,
    -1003355542910,
]

ALLOWED_CHAT_IDS = [
    -1003431402721,
    -1003355542910,
    -1003300824366,
    -1003394079022,
    -1003062290367,
]

DB_NAME = "multiverse_rp.db"

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)

BOT_STATUS_FILE = "bot_status.json"
logging_active = False
filtering_posts_active = False

def load_bot_status():
    global logging_active, filtering_posts_active
    if os.path.exists(BOT_STATUS_FILE):
        with open(BOT_STATUS_FILE, "r", encoding="utf-8") as f:
            status = json.load(f)
            logging_active = status.get("logging_active", False)
            filtering_posts_active = status.get("filtering_posts_active", False)
            logger.info(f"Загружен статус бота: logging_active={logging_active}, filtering_posts_active={filtering_posts_active}")

def save_bot_status():
    with open(BOT_STATUS_FILE, "w", encoding="utf-8") as f:
        json.dump({"logging_active": logging_active, "filtering_posts_active": filtering_posts_active}, f, indent=4, ensure_ascii=False)
    logger.info("Статус бота сохранен.")

(
    STATE_SUPPORT_MESSAGE,
    STATE_SUPPORT_REPLY,
    STATE_PLAYERBOARD_MESSAGE,
    STATE_PLAYERBOARD_ROLES,
    STATE_ADD_NAGRAD_NAME,
    STATE_ADD_NAGRAD_DESCRIPTION,
    STATE_ADD_NAGRAD_PHOTO,
    STATE_ADD_NAGRAD_COST,
    STATE_ADD_NAGRAD_TARGET_USER,
    STATE_ANKETA_MESSAGE,
    STATE_CREATE_CHECK_AMOUNT,
    STATE_CREATE_CHECK_CURRENCY,
    STATE_CREATE_CHECK_MAX_USES,
    STATE_CREATE_CHECK_PASSWORD,
    STATE_CREATE_CHECK_DESCRIPTION,
    STATE_SEND_INFO_CONTENT,
    STATE_ANKETA_CLARIFY,
) = range(17)

EMOJI_PATTERN = re.compile(
    "["
    "\U0001F600-\U0001F64F"
    "\U0001F300-\U0001F5FF"
    "\U0001F680-\U0001F6FF"
    "\U0001F1E0-\U0001F1FF"
    "\U00002702-\U000027B0"
    "\U000024C2-\U0001F251"
    "]+"
)

DATABASE_URL = os.getenv('DATABASE_URL', '').replace('postgres://', 'postgresql://')

if DATABASE_URL:
    engine = create_engine(DATABASE_URL, pool_pre_ping=True, pool_recycle=300)
    logger.info("Используется PostgreSQL база данных")
else:
    engine = create_engine(f"sqlite:///{DB_NAME}", connect_args={"check_same_thread": False})
    logger.info("Используется локальная SQLite база данных")

Base = declarative_base()
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

class StringList(TypeDecorator):
    impl = SQLA_String
    cache_ok = True

    def process_bind_param(self, value, dialect):
        if value is None:
            return json.dumps([])
        if not isinstance(value, list):
            logger.warning(f"StringList process_bind_param received non-list value: {type(value)} - {value}. Wrapping in a list.")
            value = [value]
        return json.dumps(value)

    def process_result_param(self, value, dialect):
        if value is None:
            return []
        try:
            deserialized_value = json.loads(value)
            if isinstance(deserialized_value, list):
                return deserialized_value
            else:
                logger.warning(f"StringList expected a JSON list, but got type {type(deserialized_value)} for value '{value}'. Returning empty list.")
                return []
        except json.JSONDecodeError:
            logger.error(f"StringList failed to JSON decode value: '{value}'. Returning empty list.", exc_info=True)
            return []
        except Exception as e:
            logger.error(f"Unexpected error in StringList process_result_param for value '{value}': {e}. Returning empty list.", exc_info=True)
            return []

class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, index=True)
    on_balance = Column(Integer, default=0)
    op_balance = Column(Integer, default=0)
    status_rp = Column(String, default="Участник")
    unique_code = Column(String, unique=True, index=True)
    is_developer = Column(Boolean, default=False)
    is_moderator = Column(Boolean, default=False)
    is_anketnik = Column(Boolean, default=False)
    is_banned = Column(Boolean, default=False)
    nagrads_enabled = Column(Boolean, default=True)
    show_nagrads_in_profile = Column(Boolean, default=True)
    
    roles = relationship("Role", back_populates="user", cascade="all, delete-orphan")
    created_checks = relationship("Check", back_populates="creator", cascade="all, delete-orphan")
    playerboard_entries = relationship("PlayerBoardEntry", back_populates="user", cascade="all, delete-orphan")
    message_stats = relationship("MessageStat", back_populates="user", uselist=False, cascade="all, delete-orphan")
    user_nagrads = relationship("UserNagrad", back_populates="user", foreign_keys="UserNagrad.user_id", cascade="all, delete-orphan")
    given_nagrads = relationship("UserNagrad", back_populates="given_by", foreign_keys="UserNagrad.given_by_id")
    support_requests = relationship("SupportRequest", back_populates="user", cascade="all, delete-orphan")
    posts = relationship("Post", back_populates="user", cascade="all, delete-orphan")
    anketa_requests = relationship("AnketaRequest", back_populates="user", cascade="all, delete-orphan")
    info_subscriptions = relationship("InfoSubscription", back_populates="user", uselist=False, cascade="all, delete-orphan")

    def __repr__(self):
        return f"<User(id={self.id}, username='{self.username}')>"

class Role(Base):
    __tablename__ = "roles"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    name = Column(String)
    hashtag = Column(String, index=True)
    last_active = Column(Date, default=datetime.date.today)
    last_warning_sent = Column(Date, nullable=True)

    user = relationship("User", back_populates="roles")

class Check(Base):
    __tablename__ = "checks"
    id = Column(Integer, primary_key=True, index=True)
    creator_id = Column(Integer, ForeignKey("users.id"))
    amount = Column(Integer)
    currency = Column(String, default="ON")
    description = Column(String, nullable=True)
    unique_code = Column(String, unique=True, index=True, default=lambda: str(uuid.uuid4())[:8])
    max_uses = Column(Integer, default=1)
    current_uses = Column(Integer, default=0)
    created_at = Column(DateTime, default=datetime.datetime.now)
    password = Column(String, nullable=True)

    creator = relationship("User", back_populates="created_checks")

    @property
    def is_active(self):
        return self.current_uses < self.max_uses

class PlayerBoardEntry(Base):
    __tablename__ = "player_board_entries"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    message = Column(Text)
    roles_needed = Column(StringList)
    created_at = Column(DateTime, default=datetime.datetime.now)

    user = relationship("User", back_populates="playerboard_entries")

class MessageStat(Base):
    __tablename__ = "message_stats"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), unique=True)
    message_count = Column(Integer, default=0)
    post_count = Column(Integer, default=0)
    last_updated = Column(DateTime, default=datetime.datetime.now)

    user = relationship("User", back_populates="message_stats")

class NagradDefinition(Base):
    __tablename__ = "nagrad_definitions"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, unique=True, index=True)
    description = Column(Text, nullable=True)
    photo_file_id = Column(String, nullable=True)
    cost_on = Column(Integer, default=0)

    user_nagrads = relationship("UserNagrad", back_populates="definition", cascade="all, delete-orphan")

class UserNagrad(Base):
    __tablename__ = "user_nagrads"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    nagrad_definition_id = Column(Integer, ForeignKey("nagrad_definitions.id"))
    unique_code = Column(String, unique=True, index=True, default=lambda: str(uuid.uuid4())[:8])
    created_at = Column(DateTime, default=datetime.datetime.now)
    given_by_id = Column(Integer, ForeignKey("users.id"), nullable=True)

    user = relationship("User", back_populates="user_nagrads", foreign_keys=[user_id])
    definition = relationship("NagradDefinition", back_populates="user_nagrads")
    given_by = relationship("User", back_populates="given_nagrads", foreign_keys=[given_by_id])

class SupportRequest(Base):
    __tablename__ = "support_requests"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    request_content = Column(StringList)
    status = Column(String, default="open")
    created_at = Column(DateTime, default=datetime.datetime.now)
    recipient_messages = Column(StringList, default=[])

    user = relationship("User", back_populates="support_requests")

class Post(Base):
    __tablename__ = "posts"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    content = Column(Text)
    hashtag = Column(String, index=True)
    created_at = Column(DateTime, default=datetime.datetime.now)
    message_id = Column(BigInteger, nullable=True)
    chat_id = Column(BigInteger, nullable=True)

    user = relationship("User", back_populates="posts")

class AnketaRequest(Base):
    __tablename__ = "anketa_requests"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    anketa_content = Column(StringList)
    status = Column(String, default="pending")
    created_at = Column(DateTime, default=datetime.datetime.now)
    admin_message_id = Column(BigInteger, nullable=True)
    admin_chat_id = Column(BigInteger, nullable=True)

    user = relationship("User", back_populates="anketa_requests")

class InfoSubscription(Base):
    __tablename__ = "info_subscriptions"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), unique=True)
    subscribed = Column(Boolean, default=False)

    user = relationship("User", back_populates="info_subscriptions")

def create_tables():
    Base.metadata.create_all(bind=engine)
    logger.info("Таблицы базы данных созданы или уже существуют.")
    
    session = SessionLocal()
    try:
        # Проверка и добавление колонок для PostgreSQL
        if DATABASE_URL:
            # Для PostgreSQL проверяем существование колонок
            try:
                # Проверяем существование колонки post_count
                result = session.execute(text("SELECT column_name FROM information_schema.columns WHERE table_name='message_stats' AND column_name='post_count'"))
                if not result.fetchone():
                    session.execute(text("ALTER TABLE message_stats ADD COLUMN post_count INTEGER DEFAULT 0"))
                    logger.info("Добавлена колонка post_count в таблицу message_stats.")
                
                # Проверяем существование колонки password в checks
                result = session.execute(text("SELECT column_name FROM information_schema.columns WHERE table_name='checks' AND column_name='password'"))
                if not result.fetchone():
                    session.execute(text("ALTER TABLE checks ADD COLUMN password TEXT"))
                    logger.info("Добавлена колонка password в таблице checks.")
                
                # Проверяем существование колонки last_warning_sent в roles
                result = session.execute(text("SELECT column_name FROM information_schema.columns WHERE table_name='roles' AND column_name='last_warning_sent'"))
                if not result.fetchone():
                    session.execute(text("ALTER TABLE roles ADD COLUMN last_warning_sent DATE"))
                    logger.info("Добавлена колонка last_warning_sent в таблице roles.")
                
                # Проверяем существование колонки anketa_content в anketa_requests
                result = session.execute(text("SELECT column_name FROM information_schema.columns WHERE table_name='anketa_requests' AND column_name='anketa_content'"))
                if not result.fetchone():
                    session.execute(text("ALTER TABLE anketa_requests ADD COLUMN anketa_content TEXT"))
                    logger.info("Добавлена колонка anketa_content в таблице anketa_requests.")
                
                # Проверяем существование колонки content в posts
                result = session.execute(text("SELECT column_name FROM information_schema.columns WHERE table_name='posts' AND column_name='content'"))
                if not result.fetchone():
                    session.execute(text("ALTER TABLE posts ADD COLUMN content TEXT"))
                    logger.info("Добавлена колонка content в таблице posts.")
                
                # Проверяем существование колонки is_anketnik в users
                result = session.execute(text("SELECT column_name FROM information_schema.columns WHERE table_name='users' AND column_name='is_anketnik'"))
                if not result.fetchone():
                    # Проверяем существование старой колонки is_spisochnik
                    result = session.execute(text("SELECT column_name FROM information_schema.columns WHERE table_name='users' AND column_name='is_spisochnik'"))
                    if result.fetchone():
                        session.execute(text("ALTER TABLE users RENAME COLUMN is_spisochnik TO is_anketnik"))
                        logger.info("Переименована колонка is_spisochnik в is_anketnik")
                    else:
                        session.execute(text("ALTER TABLE users ADD COLUMN is_anketnik BOOLEAN DEFAULT FALSE"))
                        logger.info("Добавлена колонка is_anketnik в таблицу users.")
                
                # Обновляем тип колонок message_id и chat_id в таблице posts на BIGINT для PostgreSQL
                result = session.execute(text("SELECT data_type FROM information_schema.columns WHERE table_name='posts' AND column_name='message_id'"))
                row = result.fetchone()
                if row and row[0] != 'bigint':
                    session.execute(text("ALTER TABLE posts ALTER COLUMN message_id TYPE BIGINT"))
                    logger.info("Исправлен тип колонки message_id на BIGINT в таблице posts.")
                
                result = session.execute(text("SELECT data_type FROM information_schema.columns WHERE table_name='posts' AND column_name='chat_id'"))
                row = result.fetchone()
                if row and row[0] != 'bigint':
                    session.execute(text("ALTER TABLE posts ALTER COLUMN chat_id TYPE BIGINT"))
                    logger.info("Исправлен тип колонки chat_id на BIGINT в таблице posts.")
                    
            except Exception as e:
                logger.error(f"Ошибка при проверке/добавлении колонок в PostgreSQL: {e}")
        else:
            # Для SQLite используем старый код
            result = session.execute(text("PRAGMA table_info(message_stats)"))
            columns = [row[1] for row in result]
            if 'post_count' not in columns:
                session.execute(text("ALTER TABLE message_stats ADD COLUMN post_count INTEGER DEFAULT 0"))
                logger.info("Добавлена колонка post_count в таблицу message_stats.")
        
        session.commit()
    except Exception as e:
        logger.error(f"Ошибка при проверке/добавлении колонок: {e}")
        session.rollback()
    finally:
        session.close()

def get_session():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def get_session_for_job():
    return SessionLocal()

def get_db_session():
    return SessionLocal()

def db_session(func):
    @wraps(func)
    async def wrapper(update: Update, context: ContextTypes.DEFAULT_TYPE, *args, **kwargs):
        session = SessionLocal()
        try:
            result = await func(update, context, session, *args, **kwargs)
            session.commit()
            return result
        except TelegramError as e:
            logger.warning(f"Telegram API error in {func.__name__}: {e}", exc_info=True)
        except Exception as e:
            session.rollback()
            logger.error(f"Unexpected error in {func.__name__}: {e}", exc_info=True)
            raise
        finally:
            session.close()
    return wrapper

def db_session_for_conversation(func):
    @wraps(func)
    async def wrapper(update: Update, context: ContextTypes.DEFAULT_TYPE, *args, **kwargs):
        session = SessionLocal()
        try:
            result = await func(update, context, session, *args, **kwargs)
            session.commit()
            return result
        except TelegramError as e:
            logger.warning(f"Telegram API error in {func.__name__}: {e}", exc_info=True)
        except Exception as e:
            session.rollback()
            logger.error(f"Unexpected error in {func.__name__}: {e}", exc_info=True)
        finally:
            session.close()
    return wrapper

def get_or_create_user(session, user_id: int, username: str | None) -> User:
    user = session.query(User).filter(User.id == user_id).first()
    if not user:
        user = User(id=user_id, username=username, unique_code=str(uuid.uuid4())[:8])
        if user_id in DEVELOPER_IDS:
            user.is_developer = True
            user.is_anketnik = True
        if user_id in INITIAL_ANKETNIK_USER_IDS:
            user.is_anketnik = True
        session.add(user)
        session.commit()
        session.refresh(user)
        logger.info(f"Новый пользователь зарегистрирован: {username} (ID: {user_id})")
    elif user.username != username:
        user.username = username
        session.commit()
        session.refresh(user)
    return user

def get_user_by_username_db(session, username: str) -> User | None:
    return session.query(User).filter(User.username == username.lstrip('@')).first()

def get_user_by_identifier_db(session, identifier: str) -> User | None:
    try:
        user_id = int(identifier)
        return session.query(User).filter(User.id == user_id).first()
    except ValueError:
        return session.query(User).filter(
            or_(User.username == identifier.lstrip('@'), User.unique_code == identifier)
        ).first()

def developer_only(func):
    @wraps(func)
    async def wrapper(update: Update, context: ContextTypes.DEFAULT_TYPE, session, *args, **kwargs):
        user_id = update.effective_user.id
        user_db = session.query(User).filter(User.id == user_id).first()
        if not user_db or not user_db.is_developer:
            if update.effective_message:
                await update.effective_message.reply_text("У вас нет прав для выполнения этой команды.")
            elif update.callback_query:
                await update.callback_query.answer("У вас нет прав для выполнения этого действия.")
            return
        return await func(update, context, session, *args, **kwargs)
    return wrapper

def moderator_or_developer_only(func):
    @wraps(func)
    async def wrapper(update: Update, context: ContextTypes.DEFAULT_TYPE, session, *args, **kwargs):
        user_id = update.effective_user.id
        user_db = session.query(User).filter(User.id == user_id).first()
        if not user_db or (not user_db.is_developer and not user_db.is_moderator):
            if update.effective_message:
                await update.effective_message.reply_text("У вас нет прав для выполнения этой команды.")
            elif update.callback_query:
                await update.callback_query.answer("У вас нет прав для выполнения этого действия.")
            return
        return await func(update, context, session, *args, **kwargs)
    return wrapper

def anketnik_or_developer_only(func):
    @wraps(func)
    async def wrapper(update: Update, context: ContextTypes.DEFAULT_TYPE, session, *args, **kwargs):
        user_id = update.effective_user.id
        user_db = session.query(User).filter(User.id == user_id).first()
        if not user_db or (not user_db.is_developer and not user_db.is_anketnik):
            if update.effective_message:
                await update.effective_message.reply_text("У вас нет прав для выполнения этой команды (только для разработчиков и Анкетников).")
            elif update.callback_query:
                await update.callback_query.answer("У вас нет прав для выполнения этого действия.")
            return
        return await func(update, context, session, *args, **kwargs)
    return wrapper

def not_banned(func):
    @wraps(func)
    async def wrapper(update: Update, context: ContextTypes.DEFAULT_TYPE, session, *args, **kwargs):
        user_id = update.effective_user.id
        user_db = session.query(User).filter(User.id == user_id).first()
        if user_db and user_db.is_banned:
            if update.effective_message:
                await update.effective_message.reply_text("Вы забанены и не можете использовать бота.")
            elif update.callback_query:
                await update.callback_query.answer("Вы забанены и не можете использовать бота.")
            return
        return await func(update, context, session, *args, **kwargs)
    return wrapper

class IsModeratorFilter(filters.BaseFilter):
    def filter(self, update: Update) -> bool:
        if not update.effective_user:
            return False
        user_id = update.effective_user.id
        session = SessionLocal()
        try:
            user_db = session.query(User).filter(User.id == user_id).first()
            return user_db and user_db.is_moderator
        finally:
            session.close()

is_moderator_filter = IsModeratorFilter()

INFO_POSTS_FILE = "info_posts.json"

def load_info_posts():
    if os.path.exists(INFO_POSTS_FILE):
        try:
            with open(INFO_POSTS_FILE, "r", encoding="utf-8") as f:
                posts = json.load(f)
                if isinstance(posts, list):
                    return posts
                else:
                    if posts:
                        return [posts]
                    else:
                        return []
        except (json.JSONDecodeError, KeyError) as e:
            logger.error(f"Ошибка при загрузке INFO постов: {e}")
            return []
    return []

def save_info_posts(posts):
    with open(INFO_POSTS_FILE, "w", encoding="utf-8") as f:
        json.dump(posts, f, ensure_ascii=False, indent=4)

async def info_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if update.callback_query:
        message = update.callback_query.message
        chat_id = message.chat_id
        await update.callback_query.answer()
    else:
        message = update.message
        chat_id = message.chat_id

    posts = load_info_posts()
    
    if not posts:
        await context.bot.send_message(chat_id=chat_id, text="/\\======I-N-F-O======/\\\n[Новостей/Событий/Рассылок пока-что нет]")
        return
    
    first_post = posts[0]
    
    caption = f"INFO пост #1 из {len(posts)}\n\n"
    
    if first_post.get('text'):
        caption += first_post['text']
    
    if 'created_at' in first_post:
        try:
            created_at = datetime.datetime.fromisoformat(first_post['created_at'])
            caption += f"\n\nДата отправления: {created_at.strftime('%d.%m.%Y %H:%M')}"
        except:
            pass
    if 'created_by' in first_post:
        caption += f"\nОтправитель: @{first_post['created_by']}"
    
    try:
        if first_post.get('photo'):
            await context.bot.send_photo(
                chat_id=chat_id,
                photo=first_post['photo'],
                caption=caption if caption.strip() else None
            )
        elif first_post.get('video'):
            await context.bot.send_video(
                chat_id=chat_id,
                video=first_post['video'],
                caption=caption if caption.strip() else None
            )
        elif first_post.get('animation'):
            await context.bot.send_animation(
                chat_id=chat_id,
                animation=first_post['animation'],
                caption=caption if caption.strip() else None
            )
        else:
            await context.bot.send_message(chat_id=chat_id, text=caption)
    except Exception as e:
        logger.error(f"Ошибка при отправке первого INFO поста: {e}")
        await context.bot.send_message(chat_id=chat_id, text=caption)
    
    if len(posts) > 1:
        list_text = f"Всего INFO постов: {len(posts)}\n\n"
        list_text += f"1. {'(Медиа)' if any(key in first_post for key in ['photo', 'video', 'animation']) else ''} {first_post.get('text', 'Без текста')[:50]}...\n"
        
        for i, post in enumerate(posts[1:], 2):
            post_type = "Текст"
            if post.get('photo'):
                post_type = "Фото"
            elif post.get('video'):
                post_type = "Видео"
            elif post.get('animation'):
                post_type = "Гифка"
            
            preview = post.get('text', 'Без текста')[:50]
            if len(preview) < len(post.get('text', '')):
                preview += "..."
            
            list_text += f"{i}. [{post_type}] {preview}\n"
        
        await context.bot.send_message(chat_id=chat_id, text=list_text)

@db_session_for_conversation
@developer_only
async def start_send_info(update: Update, context: ContextTypes.DEFAULT_TYPE, session) -> int:
    await update.message.reply_text(
        "Пожалуйста, введите ваш пост/посты в INFO. После завершения того что вы хотели написать отправьте команду /Done_info."
    )
    context.user_data['info_buffer'] = []
    return STATE_SEND_INFO_CONTENT

@db_session_for_conversation
async def send_info_content(update: Update, context: ContextTypes.DEFAULT_TYPE, session) -> int:
    message_content = {}
    
    if update.message.text:
        message_content = {'type': 'text', 'content': update.message.text}
    elif update.message.photo:
        message_content = {'type': 'photo', 'file_id': update.message.photo[-1].file_id, 'caption': update.message.caption}
    elif update.message.video:
        message_content = {'type': 'video', 'file_id': update.message.video.file_id, 'caption': update.message.caption}
    elif update.message.animation:
        message_content = {'type': 'animation', 'file_id': update.message.animation.file_id, 'caption': update.message.caption}
    else:
        await update.message.reply_text("Пожалуйста, отправляйте только текстовые сообщения, фото, видео или гифки для INFO.")
        return STATE_SEND_INFO_CONTENT

    context.user_data['info_buffer'].append(message_content)
    await update.message.reply_text("Сообщение/медиа добавлено. Продолжайте или отправьте /Done_info для завершения.")
    return STATE_SEND_INFO_CONTENT

@db_session_for_conversation
async def done_info_dialog(update: Update, context: ContextTypes.DEFAULT_TYPE, session) -> int:
    info_content_list = context.user_data.pop('info_buffer', [])
    if not info_content_list:
        await update.message.reply_text("Вы не отправили ни одного сообщения для INFO. Запрос отменен.")
        return ConversationHandler.END

    post_data = {}
    text_parts = []
    media_file_id = None
    media_type = None
    for item in info_content_list:
        if item['type'] == 'text':
            text_parts.append(item['content'])
        elif media_file_id is None:
            media_file_id = item['file_id']
            media_type = item['type']
            if item.get('caption'):
                text_parts.append(item['caption'])

    post_data['text'] = '\n\n'.join(text_parts) if text_parts else ''
    if media_file_id:
        if media_type == 'photo':
            post_data['photo'] = media_file_id
        elif media_type == 'video':
            post_data['video'] = media_file_id
        elif media_type == 'animation':
            post_data['animation'] = media_file_id

    post_data['created_at'] = datetime.datetime.now().isoformat()
    post_data['created_by'] = update.effective_user.username or update.effective_user.id

    posts = load_info_posts()
    posts.append(post_data)
    save_info_posts(posts)

    post_number = len(posts)
    
    subscriptions = session.query(InfoSubscription).filter(InfoSubscription.subscribed == True).all()
    
    caption = f"INFO пост #{post_number}\n\n{post_data.get('text', '')}\n\nДата отправления: {datetime.datetime.now().strftime('%d.%m.%Y %H:%M')}\nОтправитель: @{update.effective_user.username or update.effective_user.id}"
    
    sent_count = 0
    for sub in subscriptions:
        try:
            if 'photo' in post_data:
                await context.bot.send_photo(
                    chat_id=sub.user_id,
                    photo=post_data['photo'],
                    caption=caption
                )
            elif 'video' in post_data:
                await context.bot.send_video(
                    chat_id=sub.user_id,
                    video=post_data['video'],
                    caption=caption
                )
            elif 'animation' in post_data:
                await context.bot.send_animation(
                    chat_id=sub.user_id,
                    animation=post_data['animation'],
                    caption=caption
                )
            else:
                await context.bot.send_message(
                    chat_id=sub.user_id,
                    text=caption
                )
            sent_count += 1
        except TelegramError as e:
            logger.warning(f"Не удалось отправить INFO пост пользователю {sub.user_id}: {e}")
    
    await update.message.reply_text(f"Ваша рассылка №{post_number} была помещена на INFO. Отправлено {sent_count} пользователям.")
    return ConversationHandler.END

@db_session
@developer_only
async def delete_info_post(update: Update, context: ContextTypes.DEFAULT_TYPE, session) -> None:
    if len(context.args) < 1:
        await update.message.reply_text("Использование: /deleteinfopost [номер_поста]")
        return
    
    try:
        post_num = int(context.args[0])
        if post_num <= 0:
            await update.message.reply_text("Номер поста должен быть положительным числом.")
            return
    except ValueError:
        await update.message.reply_text("Номер поста должен быть числом.")
        return
    
    posts = load_info_posts()
    
    if post_num > len(posts):
        await update.message.reply_text(f"Пост с номером {post_num} не найден. Всего постов: {len(posts)}")
        return
    
    deleted_post = posts.pop(post_num - 1)
    save_info_posts(posts)
    
    await update.message.reply_text(f"INFO пост #{post_num} удален. Всего постов: {len(posts)}")

@db_session
@not_banned
async def info_on(update: Update, context: ContextTypes.DEFAULT_TYPE, session) -> None:
    user_tg = update.effective_user
    user_db = get_or_create_user(session, user_tg.id, user_tg.username)

    subscription = session.query(InfoSubscription).filter(InfoSubscription.user_id == user_db.id).first()
    if subscription:
        subscription.subscribed = True
    else:
        subscription = InfoSubscription(user_id=user_db.id, subscribed=True)
        session.add(subscription)

    await update.message.reply_text("Вы подписались на рассылку INFO.")

@db_session
@not_banned
async def info_off(update: Update, context: ContextTypes.DEFAULT_TYPE, session) -> None:
    user_tg = update.effective_user
    user_db = get_or_create_user(session, user_tg.id, user_tg.username)

    subscription = session.query(InfoSubscription).filter(InfoSubscription.user_id == user_db.id).first()
    if subscription:
        subscription.subscribed = False
    else:
        subscription = InfoSubscription(user_id=user_db.id, subscribed=False)
        session.add(subscription)

    await update.message.reply_text("Вы отписались от рассылки INFO.")

@db_session
@not_banned
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE, session) -> None:
    user_tg = update.effective_user
    get_or_create_user(session, user_tg.id, user_tg.username)

    keyboard = [
        [
            InlineKeyboardButton("Мой Профиль", callback_data="profile"),
            InlineKeyboardButton("Помощь", callback_data="help")
        ],
        [
            InlineKeyboardButton("Подать Анкету", callback_data="send_anketa_callback"),
            InlineKeyboardButton("Я новенький и хочу вступить", callback_data="newbie_info")
        ],
        [
            InlineKeyboardButton("Поддержка", callback_data="support_dialog"),
            InlineKeyboardButton("PLAYERBOARD", callback_data="playerboard_list")
        ],
        [
            InlineKeyboardButton("Ссылки", callback_data="links"),
            InlineKeyboardButton("INFO", callback_data="info_command")
        ]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)

    message_text = f"Здравствуй, {user_tg.first_name}! Я централизованный бот проекта Multiverse-Rp!"

    message_source = update.callback_query.message if update.callback_query else update.message

    if update.callback_query:
        try:
            await update.callback_query.edit_message_text(
                text=message_text,
                reply_markup=reply_markup
            )
            await update.callback_query.answer()
        except TelegramError as e:
            logger.warning(f"Failed to edit message for callback query (start): {e}. Falling back to reply_text.")
            await message_source.reply_text(
                text=message_text,
                reply_markup=reply_markup
            )
    else:
        await message_source.reply_text(
            text=message_text,
            reply_markup=reply_markup
        )

@db_session
@not_banned
async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE, session) -> None:
    help_message = """Доступные команды:

Общие команды:
/start - Главное меню бота.
/help - Показать это сообщение.
/profile [username/ID] - Показать информацию о профиле.
/support - Написать сообщение в поддержку.
/playerboard - Посмотреть или создать объявление для поиска партнера.
/links - Список полезных ссылок.
/checknagrad [юзернейм или уникальный код участника] - Посмотреть свои награды или награды другого пользователя.
/sellnagrad [название_награды/код_награды] - Продать свою награду за ОН.
/takecheck [код_чека] - Активировать чек.
/Nagrada_On - Включить получение и показ наград.
/Nagrada_Off - Отключить получение и показ наград.
/send [получатель/ответ на сообщение] [сумма] [ON/OP] - Перевести ОН или ОП другому пользователю.
/createcheck - Создать чек (диалог).
/listchecks - Посмотреть активные чеки.
/deletecheck [код_чека] - Удалить чек (только для создателя или разработчика).
/SendNagrada [юзернейм или уникальный код участника получателя] [название/код награды отправителя] - Отправить имеющуюся награду другому человеку.
/stata - Ваша статистика сообщений.
/allstata - Общая статистика сообщений.
/addnagrad - Добавить награду в систему (диалог).
/deletenagrad [код_награды] - Удалить свою награду.
/Nagrada [код награды] - Показать детали конкретной выданной награды.
/CheckRole [название роли/хэштег] - Проверить статус роли.
/sendanketa - Подать анкету на персонажа.
/deleteplayerboard [номер] - Удалить свою запись с PLAYERBOARD.
/Info - Показать информацию INFO.
/InfoON - Подписаться на рассылку INFO.
/InfoOFF - Отписаться от рассылки INFO.

Административные команды:
/add [@username #хэштег НазваниеРоли] - Выдать роль пользователю (можно массово через запятую).
/delete [юзернейм или уникальный код участника] [хештег] - Удалить роль у пользователя.
/check [юзернейм или уникальный код участника] - Проверить роли пользователя.
/checkpost [username/ID] [дата-дата] - Проверить статистику постов.

Команды разработчиков:
/addstatus [юзернейм или уникальный код участника] [статус] - Изменить статус РП пользователя.
/QyqyqysON [сумма] - Накрутить себе ОН.
/QyqyqysOP [сумма] - Накрутить себе ОП.
/ban [юзернейм или уникальный код участника] - Забанить пользователя.
/unban [юзернейм или уникальный код участника] - Разбанить пользователя.
/reset [юзернейм или уникальный код участника] - Обнулить балансы и награды пользователя.
/startlog, /stoplog, /filelog - Управление логированием.
/startfilter, /stopfilter - Управление фильтрацией.
/SendInfo - Отправить сообщение в INFO (диалог).
/Done_info - Завершить отправку INFO (используется внутри диалога).
/deleteinfopost [номер] - Удалить INFO пост."""
    message_source = update.callback_query.message if update.callback_query else update.message
    await message_source.reply_text(help_message)
    if update.callback_query:
        await update.callback_query.answer()

@db_session
@not_banned
async def newbie_info(update: Update, context: ContextTypes.DEFAULT_TYPE, session) -> None:
    message_text = """Здравствуй, новенький! 
Если ты хочешь вступить в наш РП проект ты должен прочитать инфо ( @MultiverseRp ) и написать анкету для своего персонажа.Напишите команду /sendAnketa чтобы увидеть шаблон анкеты и отправить ее на одобрение.После отправки ждите как вам напишут об одобрении/отказе Роли или уточнении."""
    
    message_source = update.callback_query.message if update.callback_query else update.message
    await message_source.reply_text(message_text)
    if update.callback_query:
        await update.callback_query.answer()

@db_session
@not_banned
async def profile(update: Update, context: ContextTypes.DEFAULT_TYPE, session) -> None:
    if context.args:
        target_identifier = context.args[0]
        user_db = get_user_by_identifier_db(session, target_identifier)
        if not user_db:
            if update.callback_query:
                await update.callback_query.answer(f"Пользователь '{target_identifier}' не найден.")
            else:
                await update.message.reply_text(f"Пользователь '{target_identifier}' не найден.")
            return
    else:
        user_tg = update.effective_user
        user_db = get_or_create_user(session, user_tg.id, user_tg.username)

    try:
        profile_photos = await context.bot.get_user_profile_photos(user_db.id, limit=1)
        photo_file_id = None
        if profile_photos.photos and profile_photos.photos[0]:
            photo_file_id = profile_photos.photos[0][-1].file_id
    except Exception as e:
        logger.warning(f"Не удалось получить фото профиля для пользователя {user_db.id}: {e}")
        photo_file_id = None
    
    stats = session.query(MessageStat).filter(MessageStat.user_id == user_db.id).first()
    message_count = stats.message_count if stats else 0
    post_count = stats.post_count if stats else 0

    if user_db.roles:
        roles_text = "Роли:\n"
        for role in user_db.roles:
            roles_text += f"  {role.name} - #{role.hashtag}\n"
    else:
        roles_text = "Роли: Нет ролей\n"

    profile_message = f"Профиль пользователя:\n"
    profile_message += f"ID: {user_db.id}\n"
    profile_message += f"Уникальный код: {user_db.unique_code}\n"
    profile_message += f"Username: @{user_db.username if user_db.username else 'Не указан'}\n"
    profile_message += f"Баланс ОН (Очки Наград): {user_db.on_balance}\n"
    profile_message += f"Баланс ОП (Очки Прокачки): {user_db.op_balance}\n"
    profile_message += f"Статус РП: {user_db.status_rp}\n"
    profile_message += roles_text
    profile_message += f"Всего сообщений: {message_count}\n"
    profile_message += f"Всего постов: {post_count}\n"
    
    admin_roles = []
    if user_db.is_developer:
        admin_roles.append("Разработчик")
    if user_db.is_anketnik and not user_db.is_developer:
        admin_roles.append("Анкетник")
    if user_db.is_moderator:
        admin_roles.append("Модератор")

    if admin_roles:
        profile_message += f"Административные роли: {', '.join(admin_roles)}\n"
        
    if user_db.is_banned:
        profile_message += "Статус: Забанен\n"

    keyboard = []
    if user_db.show_nagrads_in_profile:
        keyboard.append([InlineKeyboardButton("Награды пользователя", callback_data=f"show_user_nagrads_{user_db.id}")])
    keyboard.append([InlineKeyboardButton("На главную", callback_data="start")])
    reply_markup = InlineKeyboardMarkup(keyboard)

    message_source = update.callback_query.message if update.callback_query else update.message
    
    try:
        if photo_file_id and message_source.chat.type == "private":
            await context.bot.send_photo(
                chat_id=message_source.chat_id,
                photo=photo_file_id,
                caption=profile_message,
                reply_markup=reply_markup
            )
        else:
            await message_source.reply_text(profile_message, reply_markup=reply_markup)
    except Exception as e:
        logger.error(f"Ошибка при отправке профиля: {e}")
        try:
            await message_source.reply_text(profile_message, reply_markup=reply_markup)
        except:
            pass
        
    if update.callback_query:
        await update.callback_query.answer()

@db_session
@not_banned
async def show_my_nagrads(update: Update, context: ContextTypes.DEFAULT_TYPE, session) -> None:
    query = update.callback_query
    user_tg = query.from_user if query else update.effective_user
    
    if query and query.data.startswith("show_user_nagrads_"):
        try:
            target_user_id = int(query.data.split("_")[-1])
            user_db = session.query(User).filter(User.id == target_user_id).first()
            if not user_db:
                await query.answer("Пользователь не найден.")
                return
        except ValueError:
            user_db = get_or_create_user(session, user_tg.id, user_tg.username)
    else:
        user_db = get_or_create_user(session, user_tg.id, user_tg.username)
    
    if not user_db.show_nagrads_in_profile:
        if query:
            await query.answer("Пользователь отключил показ наград в профиле.")
        else:
            await update.message.reply_text("Пользователь отключил показ наград в профиле.")
        return
    
    user_nagrads = session.query(UserNagrad).filter(UserNagrad.user_id == user_db.id).all()
    
    if not user_nagrads:
        if query:
            await query.answer("У пользователя пока нет наград.")
        else:
            await update.message.reply_text("У пользователя пока нет наград.")
        return
    
    first_nagrad = user_nagrads[0]
    definition = first_nagrad.definition
    
    caption = f"Награды пользователя @{user_db.username or user_db.id} ({len(user_nagrads)} шт.):\n\n"
    caption += f"1. {definition.name}\n"
    if definition.description:
        caption += f"   Описание: {definition.description[:50]}...\n"
    caption += f"   Код: {first_nagrad.unique_code}\n"
    caption += f"   Стоимость: {definition.cost_on} ОН\n\n"
    
    if len(user_nagrads) > 1:
        caption += f"Остальные награды ({len(user_nagrads)-1} шт.):\n"
        for i, nagrad in enumerate(user_nagrads[1:11], 2):
            definition = nagrad.definition
            caption += f"{i}. {definition.name} (Код: {nagrad.unique_code})\n"
        
        if len(user_nagrads) > 11:
            caption += f"\n... и еще {len(user_nagrads) - 11} наград"
    
    if query:
        if definition.photo_file_id:
            try:
                await query.message.reply_photo(
                    photo=definition.photo_file_id,
                    caption=caption
                )
            except TelegramError:
                await query.message.reply_text(caption)
        else:
            await query.message.reply_text(caption)
        await query.answer()
    else:
        if definition.photo_file_id:
            try:
                await update.message.reply_photo(
                    photo=definition.photo_file_id,
                    caption=caption
                )
            except TelegramError:
                await update.message.reply_text(caption)
        else:
            await update.message.reply_text(caption)

@db_session
@not_banned
async def check_nagrad(update: Update, context: ContextTypes.DEFAULT_TYPE, session) -> None:
    if len(context.args) == 0:
        user_tg = update.effective_user
        user_db = get_or_create_user(session, user_tg.id, user_tg.username)
    else:
        identifier = context.args[0]
        user_db = get_user_by_identifier_db(session, identifier)
        if not user_db:
            await update.message.reply_text(f"Пользователь '{identifier}' не найден.")
            return
    
    if not user_db.show_nagrads_in_profile:
        await update.message.reply_text(f"Пользователь @{user_db.username or user_db.id} отключил показ наград.")
        return
    
    user_nagrads = session.query(UserNagrad).filter(UserNagrad.user_id == user_db.id).all()
    
    if not user_nagrads:
        await update.message.reply_text(f"У пользователя @{user_db.username or user_db.id} нет наград.")
        return
    
    first_nagrad = user_nagrads[0]
    definition = first_nagrad.definition
    
    response = f"Награды пользователя @{user_db.username or user_db.id} ({len(user_nagrads)} шт.):\n\n"
    response += f"1. {definition.name}\n"
    if definition.description:
        response += f"   Описание: {definition.description[:50]}...\n"
    response += f"   Код: {first_nagrad.unique_code}\n"
    response += f"   Стоимость: {definition.cost_on} ОН\n\n"
    
    if len(user_nagrads) > 1:
        response += f"Остальные награды ({len(user_nagrads)-1} шт.):\n"
        for i, nagrad in enumerate(user_nagrads[1:11], 2):
            definition = nagrad.definition
            response += f"{i}. {definition.name} (Код: {nagrad.unique_code})\n"
        
        if len(user_nagrads) > 11:
            response += f"\n... и еще {len(user_nagrads) - 11} наград"
    
    if definition.photo_file_id:
        try:
            await update.message.reply_photo(
                photo=definition.photo_file_id,
                caption=response
            )
        except TelegramError:
            await update.message.reply_text(response)
    else:
        await update.message.reply_text(response)

@db_session
@not_banned
async def sell_nagrad(update: Update, context: ContextTypes.DEFAULT_TYPE, session) -> None:
    if len(context.args) < 1:
        await update.message.reply_text("Использование: /sellnagrad [код_награды]")
        return
    
    nagrad_code = context.args[0]
    user_tg = update.effective_user
    user_db = get_or_create_user(session, user_tg.id, user_tg.username)
    
    user_nagrad = session.query(UserNagrad).filter(
        UserNagrad.user_id == user_db.id,
        UserNagrad.unique_code == nagrad_code
    ).first()
    
    if not user_nagrad:
        await update.message.reply_text("Награда с таким кодом не найдена.")
        return
    
    definition = user_nagrad.definition
    
    if definition.cost_on <= 0:
        await update.message.reply_text("Эту награду нельзя продать (стоимость 0 ОН).")
        return
    
    user_db.on_balance += definition.cost_on
    session.delete(user_nagrad)
    
    await update.message.reply_text(
        f"Вы успешно продали награду '{definition.name}' за {definition.cost_on} ОН.\n"
        f"Ваш баланс ОН: {user_db.on_balance}"
    )

@db_session
@not_banned
async def nagrada_on(update: Update, context: ContextTypes.DEFAULT_TYPE, session) -> None:
    user_tg = update.effective_user
    user_db = get_or_create_user(session, user_tg.id, user_tg.username)
    
    user_db.nagrads_enabled = True
    user_db.show_nagrads_in_profile = True
    
    await update.message.reply_text(
        "Получение и показ наград включены. Теперь вы можете получать награды от администраторов."
    )

@db_session
@not_banned
async def nagrada_off(update: Update, context: ContextTypes.DEFAULT_TYPE, session) -> None:
    user_tg = update.effective_user
    user_db = get_or_create_user(session, user_tg.id, user_tg.username)
    
    user_db.nagrads_enabled = False
    user_db.show_nagrads_in_profile = False
    
    await update.message.reply_text(
        "Получение и показ наград отключены. Вы не будете получать новые награды."
    )

@db_session
@not_banned
async def delete_my_nagrad(update: Update, context: ContextTypes.DEFAULT_TYPE, session) -> None:
    if len(context.args) < 1:
        await update.message.reply_text("Использование: /deletenagrad [код_награды]")
        return
    
    nagrad_code = context.args[0]
    user_tg = update.effective_user
    user_db = get_or_create_user(session, user_tg.id, user_tg.username)
    
    user_nagrad = session.query(UserNagrad).filter(
        UserNagrad.user_id == user_db.id,
        UserNagrad.unique_code == nagrad_code
    ).first()
    
    if not user_nagrad:
        await update.message.reply_text("Награда с таким кодом не найдена в вашей коллекции.")
        return
    
    nagrad_name = user_nagrad.definition.name
    session.delete(user_nagrad)
    
    await update.message.reply_text(f"Награда '{nagrad_name}' удалена из вашей коллекции.")

@db_session_for_conversation
@not_banned
async def start_add_nagrad(update: Update, context: ContextTypes.DEFAULT_TYPE, session) -> int:
    user_tg = update.effective_user
    user_db = get_or_create_user(session, user_tg.id, user_tg.username)
    
    await update.message.reply_text(
        "Начинаем создание награды. Введите название награды:"
    )
    
    context.user_data['nagrad_creator_id'] = user_db.id
    return STATE_ADD_NAGRAD_NAME

@db_session_for_conversation
async def add_nagrad_name(update: Update, context: ContextTypes.DEFAULT_TYPE, session) -> int:
    nagrad_name = update.message.text.strip()
    if not nagrad_name:
        await update.message.reply_text("Название награды не может быть пустым. Попробуйте еще раз:")
        return STATE_ADD_NAGRAD_NAME
    
    existing_nagrad = session.query(NagradDefinition).filter(NagradDefinition.name == nagrad_name).first()
    if existing_nagrad:
        await update.message.reply_text(f"Награда с названием '{nagrad_name}' уже существует. Придумайте другое название:")
        return STATE_ADD_NAGRAD_NAME
    
    context.user_data['nagrad_name'] = nagrad_name
    await update.message.reply_text(
        f"Отлично! Награда будет называться '{nagrad_name}'.\n"
        f"Теперь введите описание награды (или напишите 'нет' если описание не нужно):"
    )
    return STATE_ADD_NAGRAD_DESCRIPTION

@db_session_for_conversation
async def add_nagrad_description(update: Update, context: ContextTypes.DEFAULT_TYPE, session) -> int:
    description = update.message.text.strip()
    if description.lower() == 'нет':
        description = None
    
    context.user_data['nagrad_description'] = description
    
    await update.message.reply_text(
        "Теперь отправьте фото для награды (или напишите 'нет' если фото не нужно):"
    )
    return STATE_ADD_NAGRAD_PHOTO

@db_session_for_conversation
async def add_nagrad_photo(update: Update, context: ContextTypes.DEFAULT_TYPE, session) -> int:
    photo_file_id = None
    if update.message.photo:
        photo_file_id = update.message.photo[-1].file_id
    elif update.message.text and update.message.text.lower() == 'нет':
        photo_file_id = None
    else:
        await update.message.reply_text("Пожалуйста, отправьте фото или напишите 'нет':")
        return STATE_ADD_NAGRAD_PHOTO
    
    context.user_data['nagrad_photo'] = photo_file_id
    
    creator_id = context.user_data.get('nagrad_creator_id')
    if not creator_id:
        await update.message.reply_text("Ошибка: не найден создатель награды. Начните заново.")
        return ConversationHandler.END
    
    creator = session.query(User).filter(User.id == creator_id).first()
    if not creator:
        await update.message.reply_text("Ошибка: создатель не найден. Начните заново.")
        return ConversationHandler.END
    
    await update.message.reply_text(
        f"Теперь введите стоимость награды в ОН (Очках Наград).\n"
        f"У вас на балансе: {creator.on_balance} ОН\n"
        f"Введите число (например, 100):"
    )
    return STATE_ADD_NAGRAD_COST

@db_session_for_conversation
async def add_nagrad_cost(update: Update, context: ContextTypes.DEFAULT_TYPE, session) -> int:
    try:
        cost = int(update.message.text.strip())
        if cost <= 0:
            await update.message.reply_text("Стоимость должна быть положительным числом. Попробуйте еще раз:")
            return STATE_ADD_NAGRAD_COST
    except ValueError:
        await update.message.reply_text("Пожалуйста, введите число. Попробуйте еще раз:")
        return STATE_ADD_NAGRAD_COST
    
    creator_id = context.user_data.get('nagrad_creator_id')
    if not creator_id:
        await update.message.reply_text("Ошибка: не найден создатель награды. Начните заново.")
        return ConversationHandler.END
    
    creator = session.query(User).filter(User.id == creator_id).first()
    if not creator:
        await update.message.reply_text("Ошибка: создатель не найден. Начните заново.")
        return ConversationHandler.END
    
    if creator.on_balance < cost:
        await update.message.reply_text(
            f"У вас недостаточно ОН для создания награды такой стоимости.\n"
            f"Ваш баланс: {creator.on_balance} ОН\n"
            f"Требуется: {cost} ОН\n"
            f"Введите другую стоимость:"
        )
        return STATE_ADD_NAGRAD_COST
    
    context.user_data['nagrad_cost'] = cost
    
    await update.message.reply_text(
        "Теперь введите username или ID получателя награды (без @):"
    )
    return STATE_ADD_NAGRAD_TARGET_USER

@db_session_for_conversation
async def add_nagrad_target_user(update: Update, context: ContextTypes.DEFAULT_TYPE, session) -> int:
    target_identifier = update.message.text.strip()
    
    creator_id = context.user_data.get('nagrad_creator_id')
    if not creator_id:
        await update.message.reply_text("Ошибка: не найден создатель награды. Начните заново.")
        return ConversationHandler.END
    
    creator = session.query(User).filter(User.id == creator_id).first()
    if not creator:
        await update.message.reply_text("Ошибка: создатель не найден. Начните заново.")
        return ConversationHandler.END
    
    target_user = get_user_by_identifier_db(session, target_identifier)
    if not target_user:
        await update.message.reply_text(f"Пользователь '{target_identifier}' не найден. Попробуйте еще раз:")
        return STATE_ADD_NAGRAD_TARGET_USER
    
    if not target_user.nagrads_enabled:
        await update.message.reply_text(f"Пользователь @{target_user.username or target_user.id} отключил получение наград.")
        return ConversationHandler.END
    
    nagrad_cost = context.user_data['nagrad_cost']
    creator.on_balance -= nagrad_cost
    
    nagrad_definition = NagradDefinition(
        name=context.user_data['nagrad_name'],
        description=context.user_data.get('nagrad_description'),
        photo_file_id=context.user_data.get('nagrad_photo'),
        cost_on=nagrad_cost
    )
    session.add(nagrad_definition)
    session.flush()
    
    user_nagrad = UserNagrad(
        user_id=target_user.id,
        nagrad_definition_id=nagrad_definition.id,
        given_by_id=creator.id
    )
    session.add(user_nagrad)
    session.flush()
    
    try:
        message_text = f"Вы получили награду от @{creator.username or creator.id}!\n\n"
        message_text += f"Название: {nagrad_definition.name}\n"
        if nagrad_definition.description:
            message_text += f"Описание: {nagrad_definition.description}\n"
        message_text += f"Код награды: {user_nagrad.unique_code}\n"
        message_text += f"Стоимость: {nagrad_definition.cost_on} ОН"
        
        if nagrad_definition.photo_file_id:
            await context.bot.send_photo(
                chat_id=target_user.id,
                photo=nagrad_definition.photo_file_id,
                caption=message_text
            )
        else:
            await context.bot.send_message(
                chat_id=target_user.id,
                text=message_text
            )
    except TelegramError as e:
        logger.warning(f"Не удалось уведомить получателя {target_user.id} о награде: {e}")
    
    result_message = f"Награда '{nagrad_definition.name}' успешно создана и выдана пользователю @{target_user.username or target_user.id}!\n"
    result_message += f"Стоимость создания: {nagrad_cost} ОН\n"
    result_message += f"Ваш баланс ОН: {creator.on_balance}\n"
    result_message += f"Код награды: {user_nagrad.unique_code}"
    
    if nagrad_definition.photo_file_id:
        try:
            await update.message.reply_photo(
                photo=nagrad_definition.photo_file_id,
                caption=result_message
            )
        except TelegramError:
            await update.message.reply_text(result_message)
    else:
        await update.message.reply_text(result_message)
    
    for key in ['nagrad_creator_id', 'nagrad_name', 'nagrad_description', 'nagrad_photo', 'nagrad_cost']:
        context.user_data.pop(key, None)
    
    return ConversationHandler.END

@db_session
@not_banned
async def send_nagrada(update: Update, context: ContextTypes.DEFAULT_TYPE, session) -> None:
    if len(context.args) < 2:
        await update.message.reply_text("Использование: /SendNagrada [получатель] [код_награды]")
        return
    
    recipient_identifier = context.args[0]
    nagrad_code = context.args[1]
    
    sender_tg = update.effective_user
    sender_db = get_or_create_user(session, sender_tg.id, sender_tg.username)
    
    recipient_db = get_user_by_identifier_db(session, recipient_identifier)
    if not recipient_db:
        await update.message.reply_text(f"Получатель '{recipient_identifier}' не найден.")
        return
    
    if recipient_db.id == sender_db.id:
        await update.message.reply_text("Нельзя отправить награду самому себе.")
        return
    
    if not recipient_db.nagrads_enabled:
        await update.message.reply_text(f"Получатель @{recipient_db.username or recipient_db.id} отключил получение наград.")
        return
    
    user_nagrad = session.query(UserNagrad).filter(
        UserNagrad.user_id == sender_db.id,
        UserNagrad.unique_code == nagrad_code
    ).first()
    
    if not user_nagrad:
        await update.message.reply_text("Награда с таким кодом не найдена в вашей коллекции.")
        return
    
    user_nagrad.user_id = recipient_db.id
    user_nagrad.given_by_id = sender_db.id
    
    try:
        message_text = f"Вы получили награду от @{sender_tg.username or sender_tg.id}!\n\n"
        message_text += f"Название: {user_nagrad.definition.name}\n"
        if user_nagrad.definition.description:
            message_text += f"Описание: {user_nagrad.definition.description}\n"
        message_text += f"Код награды: {user_nagrad.unique_code}\n"
        message_text += f"Стоимость: {user_nagrad.definition.cost_on} ОН"
        
        if user_nagrad.definition.photo_file_id:
            await context.bot.send_photo(
                chat_id=recipient_db.id,
                photo=user_nagrad.definition.photo_file_id,
                caption=message_text
            )
        else:
            await context.bot.send_message(
                chat_id=recipient_db.id,
                text=message_text
            )
    except TelegramError as e:
        logger.warning(f"Не удалось уведомить получателя {recipient_db.id} о награде: {e}")
    
    await update.message.reply_text(
        f"Награда '{user_nagrad.definition.name}' успешно отправлена пользователю @{recipient_db.username or recipient_db.id}."
    )

@db_session
@not_banned
async def get_nagrada_details(update: Update, context: ContextTypes.DEFAULT_TYPE, session) -> None:
    if len(context.args) < 1:
        await update.message.reply_text("Использование: /Nagrada [код_награды]")
        return
    
    nagrad_code = context.args[0]
    
    user_nagrad = session.query(UserNagrad).filter(
        UserNagrad.unique_code == nagrad_code
    ).first()
    
    if not user_nagrad:
        await update.message.reply_text("Награда с таким кодом не найдена.")
        return
    
    definition = user_nagrad.definition
    owner = user_nagrad.user
    giver = user_nagrad.given_by
    
    response = f"Информация о награде:\n\n"
    response += f"Название: {definition.name}\n"
    if definition.description:
        response += f"Описание: {definition.description}\n"
    response += f"Код: {user_nagrad.unique_code}\n"
    response += f"Стоимость: {definition.cost_on} ОН\n"
    response += f"Выдана: {user_nagrad.created_at.strftime('%d.%m.%Y %H:%M')}\n"
    response += f"Текущий владелец: @{owner.username or owner.id}\n"
    
    if giver:
        response += f"Выдана пользователем: @{giver.username or giver.id}\n"
    
    if definition.photo_file_id:
        try:
            await update.message.reply_photo(
                photo=definition.photo_file_id,
                caption=response
            )
        except TelegramError:
            await update.message.reply_text(response)
    else:
        await update.message.reply_text(response)

@db_session
@developer_only
async def reset_user(update: Update, context: ContextTypes.DEFAULT_TYPE, session) -> None:
    if len(context.args) < 1:
        await update.message.reply_text("Использование: /reset [юзернейм или уникальный код участника]")
        return
    
    target_identifier = context.args[0]
    user_db = get_user_by_identifier_db(session, target_identifier)
    
    if not user_db:
        await update.message.reply_text(f"Пользователь '{target_identifier}' не найден.")
        return
    
    if user_db.is_developer:
        await update.message.reply_text("Нельзя сбросить разработчика.")
        return
    
    old_on = user_db.on_balance
    old_op = user_db.op_balance
    
    nagrad_count = session.query(UserNagrad).filter(UserNagrad.user_id == user_db.id).delete()
    
    user_db.on_balance = 0
    user_db.op_balance = 0
    
    user_db.status_rp = "Участник"
    
    try:
        await update.message.reply_text(
            f"Пользователь @{user_db.username or user_db.id} сброшен!\n"
            f"Удалено наград: {nagrad_count}\n"
            f"Обнулены балансы: ОН {old_on}→0, ОП {old_op}→0\n"
            f"Статус РП сброшен до: Участник"
        )
    except TimedOut:
        logger.warning(f"Timeout при отправке сообщения в reset_user, но операция выполнена")
    
    try:
        await context.bot.send_message(
            chat_id=user_db.id,
            text="Ваши данные были сброшены администратором. Все награды удалены, балансы обнулены."
        )
    except TelegramError as e:
        logger.warning(f"Не удалось уведомить пользователя {user_db.id} о сбросе: {e}")

@db_session
@not_banned
async def delete_playerboard_entry(update: Update, context: ContextTypes.DEFAULT_TYPE, session) -> None:
    if len(context.args) < 1:
        await update.message.reply_text("Использование: /deleteplayerboard [номер_записи]")
        return
    
    try:
        entry_id = int(context.args[0])
    except ValueError:
        await update.message.reply_text("Номер записи должен быть числом.")
        return
    
    entry = session.query(PlayerBoardEntry).filter(PlayerBoardEntry.id == entry_id).first()
    if not entry:
        await update.message.reply_text(f"Запись с номером {entry_id} не найдна.")
        return
    
    user_tg = update.effective_user
    user_db = get_or_create_user(session, user_tg.id, user_tg.username)
    
    if user_db.id != entry.user_id and not user_db.is_moderator and not user_db.is_developer:
        await update.message.reply_text("Вы можете удалять только свои записи.")
        return
    
    session.delete(entry)
    await update.message.reply_text(f"Запись #{entry_id} успешно удалена с PLAYERBOARD.")
    
    if user_db.id != entry.user_id:
        try:
            creator = session.query(User).filter(User.id == entry.user_id).first()
            if creator:
                await context.bot.send_message(
                    chat_id=creator.id,
                    text=f"Ваша запись на PLAYERBOARD (ID: {entry_id}) была удалена администратором @{user_tg.username or user_tg.id}."
                )
        except TelegramError as e:
            logger.warning(f"Не удалось уведомить создателя записи {entry.user_id}: {e}")

@db_session_for_conversation
@not_banned
async def send_anketa_start(update: Update, context: ContextTypes.DEFAULT_TYPE, session) -> int:
    user_tg = update.effective_user
    get_or_create_user(session, user_tg.id, user_tg.username)

    anketa_template = """Здравствуйте! Напишите анкету по этому шаблону:

Шаблон анкеты для взятия персонажа из фд:
1. Имя персонажа.
2. Вселенная персонажа.
3. Способности персонажа.
4. Какую роль вы меняете (если меняете).

Шаблон анкеты для взятия ОСА (СВОЕГО ПРИДУМАННОГО ПЕРСОНАЖА):
1. Ваш Ник.
2. Ваш Юзернейм.
3. Какую Роль вы меняете (если меняете).
4. Имя вашего персонажа.
5. Способности вашего персонажа.
6. Его характер.
7. Внешность персонажа (фото или текстовое описание).

Напишите /done_anketa когда анкета будет готова. Вашу анкету рассмотрит владелец РП."""

    message_source = update.callback_query.message if update.callback_query else update.message
    await message_source.reply_text(anketa_template)
    
    if update.callback_query:
        await update.callback_query.answer("Начинаем заполнение анкеты.")
    
    context.user_data['anketa_buffer'] = []
    return STATE_ANKETA_MESSAGE

@db_session_for_conversation
@not_banned
async def send_anketa_callback(update: Update, context: ContextTypes.DEFAULT_TYPE, session) -> int:
    query = update.callback_query
    await query.answer()
    
    user_tg = query.from_user
    get_or_create_user(session, user_tg.id, user_tg.username)

    anketa_template = """Здравствуйте! Напишите анкету по этому шаблону:

Шаблон анкеты для взятия персонажа из фд:
1. Имя персонажа.
2. Вселенная персонажа.
3. Способности персонажа.
4. Какую роль вы меняете (если меняете).

Шаблон анкеты для взятия ОСА (СВОЕГО ПРИДУМАННОГО ПЕРСОНАЖА):
1. Ваш Ник.
2. Ваш Юзернейм.
3. Какую Роль вы меняете (если меняете).
4. Имя вашего персонажа.
5. Способности вашего персонажа.
6. Его характер.
7. Внешность персонажа (фото или текстовое описание).

Напишите /done_anketa когда анкета будет готова. Вашу анкету рассмотрит владелец РП."""

    await query.message.reply_text(anketa_template)
    
    context.user_data['anketa_buffer'] = []
    return STATE_ANKETA_MESSAGE

@db_session_for_conversation
async def anketa_message(update: Update, context: ContextTypes.DEFAULT_TYPE, session) -> int:
    message_content = {}
    
    if update.message.text:
        message_content = {'type': 'text', 'content': update.message.text}
    elif update.message.photo:
        message_content = {'type': 'photo', 'file_id': update.message.photo[-1].file_id, 'caption': update.message.caption}
    elif update.message.video:
        message_content = {'type': 'video', 'file_id': update.message.video.file_id, 'caption': update.message.caption}
    elif update.message.animation:
        message_content = {'type': 'animation', 'file_id': update.message.animation.file_id, 'caption': update.message.caption}
    elif update.message.document:
        message_content = {'type': 'document', 'file_id': update.message.document.file_id, 'caption': update.message.caption}
    else:
        await update.message.reply_text("Пожалуйста, отправляйте только текстовые сообщения, фото, видео, документы или гифки для анкеты.")
        return STATE_ANKETA_MESSAGE

    context.user_data['anketa_buffer'].append(message_content)
    await update.message.reply_text("Сообщение/медиа добавлено в анкету. Продолжайте или напишите /done_anketa для завершения.")
    return STATE_ANKETA_MESSAGE

@db_session_for_conversation
async def done_anketa_dialog(update: Update, context: ContextTypes.DEFAULT_TYPE, session) -> int:
    user_tg = update.effective_user
    user_db = get_or_create_user(session, user_tg.id, user_tg.username)

    anketa_content_list = context.user_data.pop('anketa_buffer', [])
    if not anketa_content_list:
        await update.message.reply_text("Вы не отправили ни одного сообщения для анкеты. Запрос отменен.")
        return ConversationHandler.END

    new_anketa = AnketaRequest(
        user=user_db,
        anketa_content=anketa_content_list,
        status="pending"
    )
    session.add(new_anketa)
    session.commit()
    session.refresh(new_anketa)

    # Получаем всех анкетников и разработчиков
    anketniks = session.query(User).filter(
        or_(User.is_anketnik == True, User.is_developer == True)
    ).all()
    
    for anketnik in anketniks:
        try:
            # Отправляем основное сообщение с кнопками
            admin_message_text = f"Новая анкета от @{user_tg.username or user_tg.id} (ID: {user_tg.id}):"
            
            keyboard = [
                [
                    InlineKeyboardButton("Одобрить", callback_data=f"anketa_approve_{new_anketa.id}"),
                    InlineKeyboardButton("Отказать", callback_data=f"anketa_reject_{new_anketa.id}"),
                    InlineKeyboardButton("Уточнить", callback_data=f"anketa_clarify_{new_anketa.id}")
                ]
            ]
            reply_markup = InlineKeyboardMarkup(keyboard)
            
            admin_message = await context.bot.send_message(
                chat_id=anketnik.id,
                text=admin_message_text,
                reply_markup=reply_markup
            )
            
            # Отправляем контент анкеты
            for item in anketa_content_list:
                try:
                    if item['type'] == 'text':
                        await context.bot.send_message(
                            chat_id=anketnik.id,
                            text=item['content']
                        )
                    elif item['type'] == 'photo':
                        await context.bot.send_photo(
                            chat_id=anketnik.id,
                            photo=item['file_id'],
                            caption=item.get('caption', '')
                        )
                    elif item['type'] == 'video':
                        await context.bot.send_video(
                            chat_id=anketnik.id,
                            video=item['file_id'],
                            caption=item.get('caption', '')
                        )
                    elif item['type'] == 'animation':
                        await context.bot.send_animation(
                            chat_id=anketnik.id,
                            animation=item['file_id'],
                            caption=item.get('caption', '')
                        )
                    elif item['type'] == 'document':
                        await context.bot.send_document(
                            chat_id=anketnik.id,
                            document=item['file_id'],
                            caption=item.get('caption', '')
                        )
                except TelegramError as e:
                    logger.error(f"Не удалось отправить часть анкеты: {e}")
                    
        except TelegramError as e:
            logger.error(f"Не удалось отправить анкету анкетнику {anketnik.id}: {e}")

    await update.message.reply_text("Ваша анкета отправлена на рассмотрение. Ожидайте решения.")
    return ConversationHandler.END

@db_session_for_conversation
async def handle_anketa_callback(update: Update, context: ContextTypes.DEFAULT_TYPE, session) -> None:
    query = update.callback_query
    
    if not query:
        return
    
    data = query.data
    parts = data.split('_')
    action = parts[1]
    anketa_id = int(parts[2])
    
    anketa = session.query(AnketaRequest).filter(AnketaRequest.id == anketa_id).first()
    if not anketa:
        await query.answer("Анкета не найдена.")
        return
    
    user = anketa.user
    admin_username = query.from_user.username or query.from_user.id
    
    anketa_content = anketa.anketa_content
    
    if action == "approve":
        anketa.status = "approved"
        
        try:
            await context.bot.send_message(
                chat_id=user.id,
                text="Ваша анкета была одобрена!\n\nСсылка на РП:\nhttps://t.me/+3J3pcIv8wToxMDli"
            )
        except TelegramError as e:
            logger.warning(f"Не удалось уведомить пользователя {user.id} об одобрении анкеты: {e}")
        
        try:
            await context.bot.send_message(
                chat_id=ANKET_CHANNEL_ID,
                text=f"Новая анкета от @{user.username or user.id}!"
            )
            
            for item in anketa_content:
                if isinstance(item, dict):
                    if item['type'] == 'text':
                        await context.bot.send_message(
                            chat_id=ANKET_CHANNEL_ID,
                            text=item['content']
                        )
                    elif item['type'] == 'photo':
                        caption = item.get('caption', '')
                        await context.bot.send_photo(
                            chat_id=ANKET_CHANNEL_ID,
                            photo=item['file_id'],
                            caption=caption if caption else None
                        )
                    elif item['type'] == 'video':
                        caption = item.get('caption', '')
                        await context.bot.send_video(
                            chat_id=ANKET_CHANNEL_ID,
                            video=item['file_id'],
                            caption=caption if caption else None
                        )
                    elif item['type'] == 'animation':
                        caption = item.get('caption', '')
                        await context.bot.send_animation(
                            chat_id=ANKET_CHANNEL_ID,
                            animation=item['file_id'],
                            caption=caption if caption else None
                        )
                    elif item['type'] == 'document':
                        caption = item.get('caption', '')
                        await context.bot.send_document(
                            chat_id=ANKET_CHANNEL_ID,
                            document=item['file_id'],
                            caption=caption if caption else None
                        )
                else:
                    await context.bot.send_message(
                        chat_id=ANKET_CHANNEL_ID,
                        text=item
                    )
            
            await context.bot.send_message(
                chat_id=ANKET_CHANNEL_ID,
                text="Статус:✅"
            )
                
        except TelegramError as e:
            logger.error(f"Не удалось отправить анкету в канал {ANKET_CHANNEL_ID}: {e}")
            try:
                await context.bot.send_message(
                    chat_id=DEVELOPER_CHAT_ID,
                    text=f"Ошибка при отправке анкеты в канал {ANKET_CHANNEL_ID}: {e}\n\nАнкета от @{user.username or user.id} была одобрена, но не отправлена в канал."
                )
            except:
                pass
        
        new_keyboard = [
            [
                InlineKeyboardButton("Одобрено", callback_data="none"),
                InlineKeyboardButton("Отказать", callback_data="none"),
                InlineKeyboardButton("Уточнить", callback_data="none")
            ]
        ]
        
        try:
            await context.bot.edit_message_reply_markup(
                chat_id=query.message.chat_id,
                message_id=query.message.message_id,
                reply_markup=InlineKeyboardMarkup(new_keyboard)
            )
        except TelegramError as e:
            logger.warning(f"Не удалось обновить кнопки анкеты: {e}")
        
        await query.answer("Анкета одобрена и отправлена в канал.")
        
    elif action == "reject":
        anketa.status = "rejected"
        
        try:
            await context.bot.send_message(
                chat_id=user.id,
                text="Ваша анкета была отказана.\nСвяжитесь с поддержкой, нажав команду /support и высказав проблему"
            )
        except TelegramError as e:
            logger.warning(f"Не удалось уведомить пользователя {user.id} об отказе: {e}")
        
        new_keyboard = [
            [
                InlineKeyboardButton("Одобрить", callback_data="none"),
                InlineKeyboardButton("Отказано", callback_data="none"),
                InlineKeyboardButton("Уточнить", callback_data="none")
            ]
        ]
        
        try:
            await context.bot.edit_message_reply_markup(
                chat_id=query.message.chat_id,
                message_id=query.message.message_id,
                reply_markup=InlineKeyboardMarkup(new_keyboard)
            )
        except TelegramError as e:
            logger.warning(f"Не удалось обновить кнопки анкеты: {e}")
        
        await query.answer("Анкета отклонена.")
        
    elif action == "clarify":
        context.user_data['clarify_anketa_id'] = anketa_id
        context.user_data['clarify_target_user_id'] = user.id
        
        await query.message.reply_text(
            f"Вы начали диалог уточнения с пользователем @{user.username or user.id}. "
            f"Отправьте ваше сообщение для уточнения, и оно будет переслано пользователю. "
            f"Для завершения уточнения напишите /done_clarify."
        )
        
        await query.answer("Начат диалог уточнения.")
        return STATE_ANKETA_CLARIFY

@db_session_for_conversation
async def clarify_message(update: Update, context: ContextTypes.DEFAULT_TYPE, session) -> int:
    anketa_id = context.user_data.get('clarify_anketa_id')
    target_user_id = context.user_data.get('clarify_target_user_id')
    
    if not anketa_id or not target_user_id:
        await update.message.reply_text("Ошибка: данные диалога не найдены.")
        return ConversationHandler.END
    
    if update.message.text and update.message.text.startswith('/'):
        await update.message.reply_text("Пожалуйста, отправляйте только текстовые сообщения для уточнения.")
        return STATE_ANKETA_CLARIFY
    
    message_content = {}
    
    if update.message.text and not update.message.text.startswith('/'):
        message_content = {'type': 'text', 'content': update.message.text}
    elif update.message.photo:
        message_content = {'type': 'photo', 'file_id': update.message.photo[-1].file_id, 'caption': update.message.caption}
    elif update.message.video:
        message_content = {'type': 'video', 'file_id': update.message.video.file_id, 'caption': update.message.caption}
    elif update.message.document:
        message_content = {'type': 'document', 'file_id': update.message.document.file_id, 'caption': update.message.caption}
    else:
        await update.message.reply_text("Пожалуйста, отправляйте только текстовые сообщения, фото, видео или документы.")
        return STATE_ANKETA_CLARIFY
    
    try:
        admin_username = update.effective_user.username or update.effective_user.id
        
        if message_content['type'] == 'text':
            await context.bot.send_message(
                chat_id=target_user_id,
                text=f"Уточнение по вашей анкете от администратора @{admin_username}:\n\n{message_content['content']}\n\nПожалуйста, ответьте на это сообщение."
            )
        elif message_content['type'] == 'photo':
            await context.bot.send_photo(
                chat_id=target_user_id,
                photo=message_content['file_id'],
                caption=f"Уточнение по вашей анкете от администратора @{admin_username}:\n\n{message_content.get('caption', '')}\n\nПожалуйста, ответьте на это сообщение."
            )
        elif message_content['type'] == 'video':
            await context.bot.send_video(
                chat_id=target_user_id,
                video=message_content['file_id'],
                caption=f"Уточнение по вашей анкете от администратора @{admin_username}:\n\n{message_content.get('caption', '')}\n\nПожалуйста, ответьте на это сообщение."
            )
        elif message_content['type'] == 'document':
            await context.bot.send_document(
                chat_id=target_user_id,
                document=message_content['file_id'],
                caption=f"Уточнение по вашей анкете от администратора @{admin_username}:\n\n{message_content.get('caption', '')}\n\nПожалуйста, ответьте на это сообщение."
            )
        
        await update.message.reply_text("Сообщение отправлено пользователю. Ожидайте ответа.")
        
    except TelegramError as e:
        logger.error(f"Не удалось отправить уточнение пользователю {target_user_id}: {e}")
        await update.message.reply_text("Не удалось отправить сообщение. Возможно, пользователь заблокировал бота.")
    
    return STATE_ANKETA_CLARIFY

@db_session_for_conversation
async def done_clarify_dialog(update: Update, context: ContextTypes.DEFAULT_TYPE, session) -> int:
    anketa_id = context.user_data.get('clarify_anketa_id')
    
    if anketa_id:
        anketa = session.query(AnketaRequest).filter(AnketaRequest.id == anketa_id).first()
        if anketa:
            anketa.status = "clarification"
            session.commit()
    
    context.user_data.clear()
    await update.message.reply_text("Диалог уточнения завершен.")
    return ConversationHandler.END

@db_session
@not_banned
async def check_role(update: Update, context: ContextTypes.DEFAULT_TYPE, session) -> None:
    if len(context.args) < 1:
        await update.message.reply_text("Использование: /CheckRole [название роли/хэштег]")
        return
    
    role_identifier = " ".join(context.args).strip().lstrip('#')
    
    role = session.query(Role).filter(
        or_(func.lower(Role.hashtag) == role_identifier.lower(), Role.name.ilike(f"%{role_identifier}%"))
    ).first()
    
    if not role:
        await update.message.reply_text(f"Роль '{role_identifier}' не найдена.")
        return
    
    owner = session.query(User).filter(User.id == role.user_id).first()
    if not owner:
        await update.message.reply_text("Ошибка: владелец роли не найден.")
        return
    
    today = datetime.date.today()
    days_inactive = (today - role.last_active).days
    
    response = f"Роль занята!\n"
    response += f"Владелец Роли: @{owner.username or owner.id}\n"
    response += f"Название роли: {role.name}\n"
    response += f"Хэштег: #{role.hashtag}\n"
    response += f"Взял Роль: {role.last_active.strftime('%d.%m.%Y')}\n"
    response += f"Неактив на Роли: {days_inactive} дней"
    
    await update.message.reply_text(response)

@db_session
@developer_only
async def qyqyqys_on(update: Update, context: ContextTypes.DEFAULT_TYPE, session) -> None:
    if len(context.args) < 1:
        await update.message.reply_text("Использование: /QyqyqysON [сумма]")
        return
    
    try:
        amount = int(context.args[0])
        if amount <= 0:
            await update.message.reply_text("Сумма должна быть положительным числом.")
            return
    except ValueError:
        await update.message.reply_text("Пожалуйста, введите число.")
        return
    
    user_tg = update.effective_user
    user_db = get_or_create_user(session, user_tg.id, user_tg.username)
    
    user_db.on_balance += amount
    
    await update.message.reply_text(
        f"Вам начислено {amount} ОН.\n"
        f"Ваш баланс ОН: {user_db.on_balance}"
    )

@db_session
@developer_only
async def qyqyqys_op(update: Update, context: ContextTypes.DEFAULT_TYPE, session) -> None:
    if len(context.args) < 1:
        await update.message.reply_text("Использование: /QyqyqysOP [сумма]")
        return
    
    try:
        amount = int(context.args[0])
        if amount <= 0:
            await update.message.reply_text("Сумма должна быть положительным числом.")
            return
    except ValueError:
        await update.message.reply_text("Пожалуйста, введите число.")
        return
    
    user_tg = update.effective_user
    user_db = get_or_create_user(session, user_tg.id, user_tg.username)
    
    user_db.op_balance += amount
    
    await update.message.reply_text(
        f"Вам начислено {amount} ОП.\n"
        f"Ваш баланс ОП: {user_db.op_balance}"
    )

@db_session_for_conversation
@not_banned
async def start_support_dialog(update: Update, context: ContextTypes.DEFAULT_TYPE, session) -> int:
    message_source = update.callback_query.message if update.callback_query else update.message
    
    if message_source.chat.type != "private" and message_source.chat.id not in ALLOWED_CHAT_IDS:
        await message_source.reply_text("Бот не работает в этом чате. Используйте его в разрешенных группах или в личных сообщениях.")
        if update.callback_query:
            await update.callback_query.answer()
        return ConversationHandler.END

    user_tg = update.effective_user
    get_or_create_user(session, user_tg.id, user_tg.username)

    await message_source.reply_text("Напишите ваше сообщение в поддержку (текст, фото, видео, гифки). Когда закончите, напишите /done_support.")
    if update.callback_query:
        await update.callback_query.answer("Начата отправка сообщения в поддержку.")
    
    context.user_data['support_buffer'] = []
    return STATE_SUPPORT_MESSAGE

@db_session_for_conversation
async def support_message(update: Update, context: ContextTypes.DEFAULT_TYPE, session) -> int:
    if update.effective_chat.type != "private" and update.effective_chat.id not in ALLOWED_CHAT_IDS:
        await update.message.reply_text("Бот не работает в этом чате. Используйте его в разрешенных группах или в личных сообщениях.")
        return ConversationHandler.END

    message_content = {}
    
    if update.message.text:
        message_content = {'type': 'text', 'content': update.message.text}
    elif update.message.photo:
        message_content = {'type': 'photo', 'file_id': update.message.photo[-1].file_id, 'caption': update.message.caption}
    elif update.message.video:
        message_content = {'type': 'video', 'file_id': update.message.video.file_id, 'caption': update.message.caption}
    elif update.message.animation:
        message_content = {'type': 'animation', 'file_id': update.message.animation.file_id, 'caption': update.message.caption}
    elif update.message.document:
        message_content = {'type': 'document', 'file_id': update.message.document.file_id, 'caption': update.message.caption}
    else:
        await update.message.reply_text("Пожалуйста, отправляйте только текстовые сообщения, фото, видео, документы или гифки для поддержки.")
        return STATE_SUPPORT_MESSAGE

    context.user_data['support_buffer'].append(message_content)
    await update.message.reply_text("Сообщение/медиа добавлено. Продолжайте или напишите /done_support для завершения.")
    return STATE_SUPPORT_MESSAGE

@db_session_for_conversation
async def done_support_dialog(update: Update, context: ContextTypes.DEFAULT_TYPE, session) -> int:
    if update.effective_chat.type != "private" and update.effective_chat.id not in ALLOWED_CHAT_IDS:
        await update.message.reply_text("Бот не работает в этом чате. Используйте его в разрешенных группах или в личных сообщениях.")
        return ConversationHandler.END

    user_tg = update.effective_user
    user_db = get_or_create_user(session, user_tg.id, user_tg.username)

    support_content_list = context.user_data.pop('support_buffer', [])
    if not support_content_list:
        await update.message.reply_text("Вы не отправили ни одного сообщения или медиа для поддержки. Запрос отменен.")
        return ConversationHandler.END

    keyboard = [
        [
            InlineKeyboardButton("Ответить", callback_data=f"support_reply_{user_tg.id}"),
            InlineKeyboardButton("Завершить", callback_data=f"support_end_dialog_{user_tg.id}"),
        ]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)

    recipient_chats = []
    if DEVELOPER_CHAT_ID != 0:
        recipient_chats.append(DEVELOPER_CHAT_ID)

    if not recipient_chats:
        await update.message.reply_text("Ошибка: Не настроены ID чатов для отправки запросов поддержки. Обратитесь к администратору бота.")
        return ConversationHandler.END
    
    sent_main_messages_info = []

    for chat_id in recipient_chats:
        try:
            main_message_text = f"Новый запрос в поддержку от @{user_tg.username or user_tg.id} (ID: {user_tg.id}):"
            admin_message = await context.bot.send_message(
                chat_id=chat_id,
                text=main_message_text,
                reply_markup=reply_markup
            )
            sent_main_messages_info.append({'chat_id': chat_id, 'message_id': admin_message.message_id})
            logger.info(f"Основное сообщение запроса поддержки от {user_tg.username} отправлено в чат {chat_id}.")

            for item in support_content_list:
                try:
                    if item['type'] == 'text':
                        await context.bot.send_message(
                            chat_id=chat_id,
                            text=item['content']
                        )
                    elif item['type'] == 'photo':
                        await context.bot.send_photo(
                            chat_id=chat_id,
                            photo=item['file_id'],
                            caption=item.get('caption', '')
                        )
                    elif item['type'] == 'video':
                        await context.bot.send_video(
                            chat_id=chat_id,
                            video=item['file_id'],
                            caption=item.get('caption', '')
                        )
                    elif item['type'] == 'animation':
                        await context.bot.send_animation(
                            chat_id=chat_id,
                            animation=item['file_id'],
                            caption=item.get('caption', '')
                        )
                    elif item['type'] == 'document':
                        await context.bot.send_document(
                            chat_id=chat_id,
                            document=item['file_id'],
                            caption=item.get('caption', '')
                        )
                except TelegramError as e:
                    logger.error(f"Не удалось отправить часть запроса поддержки (тип: {item['type']}) в чат {chat_id}: {e}")

        except TelegramError as e:
            logger.error(f"Не удалось отправить основное сообщение запроса поддержки в чат {chat_id}: {e}")

    if sent_main_messages_info:
        new_support_request = SupportRequest(
            user=user_db,
            request_content=support_content_list,
            status="open",
            recipient_messages=sent_main_messages_info
        )
        session.add(new_support_request)
        await update.message.reply_text("Ваш запрос в поддержку отправлен. Ожидайте ответа.")
        return ConversationHandler.END
    else:
        await update.message.reply_text(
            "Произошла ошибка при отправке запроса поддержки. Пожалуйста, попробуйте позже или свяжитесь с администрацией."
        )
        return ConversationHandler.END

@db_session_for_conversation
@moderator_or_developer_only
async def handle_support_callback(update: Update, context: ContextTypes.DEFAULT_TYPE, session) -> None:
    query = update.callback_query
    
    if not query:
        logger.error("handle_support_callback called without a callback_query.")
        return

    data = query.data
    parts = data.split('_')
    action = parts[1]
    target_user_id = int(parts[3])

    target_user = session.query(User).filter(User.id == target_user_id).first()
    if not target_user:
        await query.answer("Пользователь не найден в базе данных.")
        return

    support_request = session.query(SupportRequest).filter(SupportRequest.user_id == target_user_id, SupportRequest.status.in_(['open', 'replied'])).order_by(SupportRequest.created_at.desc()).first()
    if not support_request:
        await query.answer("Активный запрос поддержки от этого пользователя не найден.")
        try:
            await query.message.edit_reply_markup(reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("Запрос уже обработан", callback_data="none")]]))
        except TelegramError as e:
            logger.warning(f"Could not edit message for processed support request: {e}")
        return

    admin_username = query.from_user.username or query.from_user.id
    
    new_keyboard_for_admin_chat = [
        [
            InlineKeyboardButton("Ответить", callback_data=f"support_reply_{target_user_id}"),
            InlineKeyboardButton("Завершить", callback_data=f"support_end_dialog_{target_user_id}"),
        ]
    ]

    if action == "end":
        support_request.status = "closed"
        response_text = f"Ваш запрос в поддержку был закрыт администратором @{admin_username}."
        await query.answer("Запрос поддержки закрыт.")
        new_keyboard_for_admin_chat[0][0] = InlineKeyboardButton("Ответить", callback_data="none")
        new_keyboard_for_admin_chat[0][1] = InlineKeyboardButton(f"Статус: Закрыто", callback_data="none")

        try:
            await context.bot.send_message(
                chat_id=target_user_id,
                text=response_text
            )
        except TelegramError as e:
            logger.warning(f"Не удалось уведомить пользователя {target_user_id} о закрытии запроса поддержки: {e}")

        for msg_info in support_request.recipient_messages:
            if not isinstance(msg_info, dict):
                logger.error(f"Unexpected non-dict item found in support_request.recipient_messages for request {support_request.id} (status: {support_request.status}): {msg_info}. Skipping edit.")
                continue
            
            try:
                await context.bot.edit_message_reply_markup(
                    chat_id=msg_info['chat_id'],
                    message_id=msg_info['message_id'],
                    reply_markup=InlineKeyboardMarkup(new_keyboard_for_admin_chat)
                )
            except TelegramError as e:
                logger.warning(f"Не удалось обновить сообщение поддержки в чате {msg_info['chat_id']}: {e}")
    else:
        await query.answer("Неизвестное действие.")

@db_session_for_conversation
@moderator_or_developer_only
async def start_reply_to_support(update: Update, context: ContextTypes.DEFAULT_TYPE, session) -> int:
    query = update.callback_query
    
    if query and query.data.startswith("support_reply_"):
        target_user_id = int(query.data.split('_')[2])
    else:
        logger.error("start_reply_to_support called without a proper callback query.")
        if query:
            await query.answer("Ошибка при определении пользователя для ответа. Попробуйте снова.")
        return ConversationHandler.END

    target_user = session.query(User).filter(User.id == target_user_id).first()
    if not target_user:
        if query:
            await query.answer("Пользователь не найден.")
        else:
            await update.message.reply_text("Пользователь не найден.")
        return ConversationHandler.END

    await (query.message.reply_text if query else update.message.reply_text)(
        f"Вы отвечаете пользователю @{target_user.username or target_user.id}. Введите ваше сообщение:"
    )
    if query:
        await query.answer()

    context.user_data['reply_target_user_id'] = target_user_id
    context.user_data['reply_type'] = 'support'

    support_request = session.query(SupportRequest).filter(SupportRequest.user_id == target_user_id, SupportRequest.status.in_(['open', 'replied'])).order_by(SupportRequest.created_at.desc()).first()
    if support_request:
        context.user_data['original_support_message_info'] = support_request.recipient_messages
    else:
        context.user_data['original_support_message_info'] = []

    return STATE_SUPPORT_REPLY

@db_session_for_conversation
async def support_reply_message(update: Update, context: ContextTypes.DEFAULT_TYPE, session) -> int:
    if update.effective_chat.type != "private" and update.effective_chat.id not in ALLOWED_CHAT_IDS:
        await update.message.reply_text("Бот не работает в этом чате. Используйте его в разрешенных группах или в личных сообщениях.")
        return ConversationHandler.END

    reply_text = update.message.text
    target_user_id = context.user_data.pop('reply_target_user_id', None)
    
    if not target_user_id:
        await update.message.reply_text("Ошибка: Не удалось определить получателя. Пожалуйста, начните заново.")
        return ConversationHandler.END

    target_user = session.query(User).filter(User.id == target_user_id).first()
    if not target_user:
        await update.message.reply_text("Ошибка: Пользователь не найден.")
        return ConversationHandler.END

    admin_username = update.effective_user.username or update.effective_user.id
    full_reply_text = f"Ответ по вашему запросу в поддержку от администратора @{admin_username}:\n\n{reply_text}"

    try:
        await context.bot.send_message(chat_id=target_user_id, text=full_reply_text)
        await update.message.reply_text("Ваш ответ отправлен пользователю.")

        support_request = session.query(SupportRequest).filter(SupportRequest.user_id == target_user_id, SupportRequest.status.in_(['open', 'replied'])).order_by(SupportRequest.created_at.desc()).first()
        if support_request:
            support_request.status = "replied"
            keyboard = [
                [
                    InlineKeyboardButton("Ответить (Отвечено)", callback_data=f"support_reply_{target_user_id}"),
                    InlineKeyboardButton("Завершить", callback_data=f"support_end_dialog_{target_user_id}"),
                ]
            ]
            reply_markup = InlineKeyboardMarkup(keyboard)

            for msg_info in context.user_data.pop('original_support_message_info', []):
                if not isinstance(msg_info, dict):
                    logger.error(f"Unexpected non-dict item found in support_request.recipient_messages: {msg_info}. Skipping edit.")
                    continue
                try:
                    await context.bot.edit_message_reply_markup(
                        chat_id=msg_info['chat_id'],
                        message_id=msg_info['message_id'],
                        reply_markup=reply_markup
                    )
                except TelegramError as e:
                    logger.warning(f"Failed to update support message reply markup in chat {msg_info['chat_id']}: {e}")

    except TelegramError as e:
        logger.error(f"Не удалось отправить ответ пользователю {target_user_id}: {e}")
        await update.message.reply_text("Не удалось отправить ответ пользователю. Возможно, он заблокировал бота.")
    finally:
        context.user_data.pop('reply_type', None)
    return ConversationHandler.END

@db_session_for_conversation
@not_banned
async def log_and_stats_message_handler(update: Update, context: ContextTypes.DEFAULT_TYPE, session) -> None:
    user_tg = update.effective_user
    user_db = get_or_create_user(session, user_tg.id, user_tg.username)

    stats = session.query(MessageStat).filter(MessageStat.user_id == user_db.id).first()
    if not stats:
        stats = MessageStat(user=user_db, message_count=0, post_count=0)
        session.add(stats)
    
    stats.message_count += 1
    stats.last_updated = datetime.datetime.now()

    message_text = update.effective_message.text or update.effective_message.caption or ""
    
    should_log = update.effective_chat.id in LOGGING_CHAT_IDS
    
    is_post = False
    
    # Поиск хэштега в сообщении
    hashtag = None
    if message_text:
        words = message_text.split()
        for word in words:
            if word.startswith('#'):
                hashtag = word[1:]  # Убираем #
                break
    
    if hashtag:
        post_text = message_text.strip()
        
        has_hashtag = True
        has_min_length = len(post_text) >= 3
        has_no_special_chars = True
        
        emoji_count = len(EMOJI_PATTERN.findall(post_text))
        has_max_emoji = emoji_count < 100
        
        has_media = (update.effective_message.photo is not None or 
                    update.effective_message.video is not None or
                    update.effective_message.animation is not None)
        
        user_has_role = False
        try:
            user_role = session.query(Role).filter(
                Role.user_id == user_db.id,
                func.lower(Role.hashtag) == func.lower(hashtag)
            ).first()
            user_has_role = user_role is not None
        except Exception as e:
            logger.error(f"Ошибка при проверке роли: {e}")
            user_has_role = False
        
        if user_has_role and (has_media or (has_hashtag and has_min_length and has_max_emoji and has_no_special_chars)):
            try:
                new_post = Post(
                    user=user_db,
                    content=post_text,
                    hashtag=hashtag,
                    message_id=update.effective_message.message_id,
                    chat_id=update.effective_chat.id
                )
                session.add(new_post)
                
                stats.post_count += 1
                
                if user_role:
                    user_role.last_active = datetime.date.today()
                    user_role.last_warning_sent = None
                    logger.debug(f"Активность роли '{user_role.name}' обновлена для пользователя {user_db.username}")
                
                is_post = True
                
                if should_log and logging_active:
                    log_entry = (
                        f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] "
                        f"[POST] Chat ID: {update.effective_chat.id}, "
                        f"User ID: {user_tg.id}, "
                        f"Username: @{user_tg.username or user_tg.id}, "
                        f"Hashtag: #{hashtag}, "
                        f"Message: {post_text[:100]}\n"
                    )
                    with open("log.txt", "a", encoding="utf-8") as f:
                        f.write(log_entry)
            except Exception as e:
                logger.error(f"Ошибка при сохранении поста: {e}")
    
    if should_log and logging_active and not is_post and message_text:
        log_entry = (
            f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] "
            f"Chat ID: {update.effective_chat.id}, "
            f"User ID: {user_tg.id}, "
            f"Username: @{user_tg.username or user_tg.id}, "
            f"Message: {message_text[:100] or 'FILE/PHOTO'}\n"
        )
        with open("log.txt", "a", encoding="utf-8") as f:
            f.write(log_entry)
    
    session.commit()

@db_session
@moderator_or_developer_only
async def check_post_stats(update: Update, context: ContextTypes.DEFAULT_TYPE, session) -> None:
    if len(context.args) == 0:
        user_tg = update.effective_user
        user_db = get_or_create_user(session, user_tg.id, user_tg.username)
        target_user = user_db
        date_range = None
    elif len(context.args) == 1:
        target_identifier = context.args[0]
        target_user = get_user_by_identifier_db(session, target_identifier)
        if not target_user:
            await update.message.reply_text(f"Пользователь '{target_identifier}' не найден.")
            return
        date_range = None
    elif len(context.args) >= 2:
        target_identifier = context.args[0]
        target_user = get_user_by_identifier_db(session, target_identifier)
        if not target_user:
            await update.message.reply_text(f"Пользователь '{target_identifier}' не найден.")
            return
        
        try:
            date_str = " ".join(context.args[1:])
            if "-" in date_str:
                start_str, end_str = date_str.split("-", 1)
                start_date = datetime.datetime.strptime(start_str.strip(), "%d.%m.%Y").date()
                end_date = datetime.datetime.strptime(end_str.strip(), "%d.%m.%Y").date()
                date_range = (start_date, end_date)
            else:
                single_date = datetime.datetime.strptime(date_str.strip(), "%d.%m.%Y").date()
                date_range = (single_date, single_date)
        except ValueError:
            await update.message.reply_text("Неверный формат даты. Используйте ДД.ММ.ГГГГ или ДД.ММ.ГГГГ-ДД.ММ.ГГГГ")
            return
    else:
        await update.message.reply_text("Использование: /checkpost [username/ID] [дата-дата]")
        return
    
    stats = session.query(MessageStat).filter(MessageStat.user_id == target_user.id).first()
    message_count = stats.message_count if stats else 0
    post_count = stats.post_count if stats else 0
    
    query = session.query(Post).filter(Post.user_id == target_user.id)
    
    if date_range:
        start_date, end_date = date_range
        start_datetime = datetime.datetime.combine(start_date, datetime.time.min)
        end_datetime = datetime.datetime.combine(end_date, datetime.time.max)
        query = query.filter(Post.created_at.between(start_datetime, end_datetime))
        date_text = f"с {start_date.strftime('%d.%m.%Y')} по {end_date.strftime('%d.%m.%Y')}"
    else:
        date_text = "за все время"
    
    posts = query.order_by(Post.created_at.desc()).all()
    period_post_count = len(posts)
    
    last_post = posts[0] if posts else None
    
    response = f"Статистика постов для пользователя @{target_user.username or target_user.id} {date_text}:\n\n"
    response += f"Количество постов: {period_post_count}\n"
    response += f"Всего сообщений: {message_count}\n"
    response += f"Всего постов за все время: {post_count}\n"
    
    if last_post:
        response += f"Последний пост: {last_post.created_at.strftime('%d.%m.%Y %H:%M:%S')}\n"
        response += f"Хэштег последнего поста: #{last_post.hashtag}\n"
    else:
        response += "Последний пост: нет постов\n"
    
    await update.message.reply_text(response)

async def check_inactive_roles_with_warnings(context: ContextTypes.DEFAULT_TYPE) -> None:
    session = get_session_for_job()
    try:
        today = datetime.date.today()
        
        warning_threshold = today - datetime.timedelta(days=7)
        removal_threshold = today - datetime.timedelta(days=10)
        
        roles_to_warn = session.query(Role).filter(
            Role.last_active < warning_threshold,
            Role.last_active >= removal_threshold,
            or_(Role.last_warning_sent.is_(None), Role.last_warning_sent < warning_threshold)
        ).all()
        
        for role in roles_to_warn:
            user = session.query(User).filter(User.id == role.user_id).first()
            if user:
                try:
                    await context.bot.send_message(
                        chat_id=user.id,
                        text=f"Внимание! Ваша роль '{role.name}' (хэштег #{role.hashtag}) неактивна более 7 дней. "
                             f"Если вы не сделаете пост с этим хэштегом в течение 3 дней, роль будет автоматически удалена."
                    )
                    role.last_warning_sent = today
                    logger.info(f"Отправлено предупреждение пользователю {user.id} о неактивной роли {role.name}")
                except TelegramError as e:
                    logger.warning(f"Не удалось отправить предупреждение пользователю {user.id}: {e}")
        
        roles_to_remove = session.query(Role).filter(Role.last_active < removal_threshold).all()
        
        for role in roles_to_remove:
            user = session.query(User).filter(User.id == role.user_id).first()
            role_name = role.name
            hashtag = role.hashtag
            
            session.delete(role)
            
            if user:
                try:
                    await context.bot.send_message(
                        chat_id=user.id,
                        text=f"Ваша роль '{role_name}' (хэштег #{hashtag}) была удалена за неактивность более 10 дней."
                    )
                    logger.info(f"Удалена роль '{role_name}' у пользователя {user.id} за неактивность")
                except TelegramError as e:
                    logger.warning(f"Не удалось уведомить пользователя {user.id} об удалении роли: {e}")
        
        session.commit()
        
    except Exception as e:
        logger.error(f"Ошибка при выполнении задачи check_inactive_roles_with_warnings: {e}", exc_info=True)
    finally:
        session.close()

@db_session
@not_banned
async def playerboard_list(update: Update, context: ContextTypes.DEFAULT_TYPE, session) -> None:
    entries = session.query(PlayerBoardEntry).order_by(PlayerBoardEntry.created_at.desc()).limit(10).all()
    
    if not entries:
        message_text = "PLAYERBOARD пуст. Будьте первым, кто ищет партнера!"
        keyboard = [
            [InlineKeyboardButton("Создать запись", callback_data="playerboard_create")],
            [InlineKeyboardButton("На главную", callback_data="start")]
        ]
    else:
        message_text = "Актуальные записи на PLAYERBOARD:\n\n"
        keyboard_entries = []
        for i, entry in enumerate(entries):
            user = session.query(User).filter(User.id == entry.user_id).first()
            username = user.username if user else f"ID:{entry.user_id}"
            
            roles_str = ""
            if entry.roles_needed and isinstance(entry.roles_needed, list):
                valid_roles = [role for role in entry.roles_needed if role and isinstance(role, str) and role.strip()]
                if valid_roles:
                    roles_str = ", ".join(valid_roles)
            
            message_text += f"{i+1}. От @{username} ({entry.created_at.strftime('%d.%m %H:%M')}):\n"
            message_text += f"   Сообщение: {entry.message[:100]}...\n"
            if roles_str:
                message_text += f"   Ищет роли: {roles_str}\n"
            else:
                message_text += f"   Ищет роли: Не указаны\n"
            message_text += f"   [Связаться: /player_contact_{entry.user_id}]\n\n"
            
            keyboard_entries.append([InlineKeyboardButton(f"Пригласить на РП {i+1} человека из списка", callback_data=f"player_invite_{entry.id}")])
        
        keyboard = keyboard_entries + [
            [InlineKeyboardButton("Создать запись", callback_data="playerboard_create")],
            [InlineKeyboardButton("На главную", callback_data="start")]
        ]

    reply_markup = InlineKeyboardMarkup(keyboard)
    
    message_source = update.callback_query.message if update.callback_query else update.message
    
    if update.callback_query:
        try:
            await update.callback_query.edit_message_text(
                text=message_text,
                reply_markup=reply_markup
            )
            await update.callback_query.answer()
        except TelegramError as e:
            logger.warning(f"Failed to edit message for playerboard_list callback query: {e}. Falling back to reply_text.")
            await message_source.reply_text(
                text=message_text,
                reply_markup=reply_markup
            )
    else:
        await message_source.reply_text(
            text=message_text,
            reply_markup=reply_markup
        )

@db_session_for_conversation
@not_banned
async def start_player_dialog(update: Update, context: ContextTypes.DEFAULT_TYPE, session) -> int:
    message_source = update.callback_query.message if update.callback_query else update.message
    await message_source.reply_text(
        "Начнем создание записи для PLAYERBOARD. Введите ваше сообщение для объявления (например, 'Ищу партнера для темной фентези'):"
    )
    if update.callback_query:
        await update.callback_query.answer("Начинается создание записи на PLAYERBOARD.")
    
    context.user_data['player_data'] = {}
    return STATE_PLAYERBOARD_MESSAGE

@db_session_for_conversation
async def player_message_step(update: Update, context: ContextTypes.DEFAULT_TYPE, session) -> int:
    message_text = update.message.text.strip()
    if not message_text:
        await update.message.reply_text("Сообщение не может быть пустым. Пожалуйста, введите ваше объявление.")
        return STATE_PLAYERBOARD_MESSAGE
    
    context.user_data['player_data']['message'] = message_text
    await update.message.reply_text("Отлично! Теперь укажите через запятую хэштеги ролей, которые вы ищете (например, #Маг, #Воин). Если роли не важны, напишите 'нет':")
    return STATE_PLAYERBOARD_ROLES

@db_session_for_conversation
async def player_roles_step(update: Update, context: ContextTypes.DEFAULT_TYPE, session) -> int:
    roles_input = update.message.text.strip()
    roles_needed = []
    if roles_input.lower() != 'нет':
        roles_list = roles_input.split(',')
        for role in roles_list:
            role_clean = role.strip().lstrip('#').strip()
            if role_clean:
                roles_needed.append(role_clean)

    user_tg = update.effective_user
    user_db = get_or_create_user(session, user_tg.id, user_tg.username)

    new_player_entry = PlayerBoardEntry(
        user=user_db,
        message=context.user_data['player_data']['message'],
        roles_needed=roles_needed
    )
    session.add(new_player_entry)
    
    await update.message.reply_text("Ваша запись на PLAYERBOARD успешно создана! Она будет видна в списке.")
    context.user_data.pop('player_data')
    return ConversationHandler.END

@db_session
@not_banned
async def player_contact(update: Update, context: ContextTypes.DEFAULT_TYPE, session) -> None:
    if not context.matches:
        await update.message.reply_text("Ошибка в команде. Используйте формат /player_contact_[ID пользователя].")
        return

    match = context.matches[0]
    target_user_id_str = match.group(1)
    
    try:
        target_user_id = int(target_user_id_str)
    except ValueError:
        await update.message.reply_text("Неверный ID пользователя в команде.")
        return

    target_user_db = session.query(User).filter(User.id == target_user_id).first()
    if not target_user_db:
        await update.message.reply_text("Пользователь, создавший объявление, не найден.")
        return
    
    sender_tg = update.effective_user
    sender_db = get_or_create_user(session, sender_tg.id, sender_tg.username)

    if sender_db.id == target_user_db.id:
        await update.message.reply_text("Вы пытаетесь связаться с самим собой.")
        return

    message_text = f"Пользователь @{sender_tg.username or sender_tg.id} заинтересовался вашим объявлением на PLAYERBOARD и хочет связаться с вами!\n\n"
    if sender_tg.username:
        message_text += f"Вы можете написать ему в личку: https://t.me/{sender_tg.username}"
    else:
        message_text += f"Вы можете написать ему в личку по его ID: @{sender_tg.id}"
    
    try:
        await context.bot.send_message(chat_id=target_user_db.id, text=message_text)
        await update.message.reply_text(f"Пользователю @{target_user_db.username or target_user_db.id} отправлено уведомление о вашем желании связаться. Он сам напишет вам, если будет заинтересован.")
    except TelegramError as e:
        logger.warning(f"Не удалось отправить сообщение пользователю {target_user_db.id}: {e}")
        await update.message.reply_text(f"Не удалось отправить уведомление пользователю @{target_user_db.username or target_user_db.id}. Возможно, он заблокировал бота. Вы можете попробовать написать ему в личку, если его никнейм публичный: @{target_user_db.username}")

@db_session_for_conversation
@not_banned
async def handle_player_invite_callback(update: Update, context: ContextTypes.DEFAULT_TYPE, session) -> None:
    query = update.callback_query
    await query.answer()
    
    data = query.data
    parts = data.split('_')
    entry_id = int(parts[2])
    inviter_id = query.from_user.id

    entry = session.query(PlayerBoardEntry).filter(PlayerBoardEntry.id == entry_id).first()
    if not entry:
        await query.message.reply_text("Эта запись на PLAYERBOARD больше не существует.")
        return
    
    entry_owner_db = session.query(User).filter(User.id == entry.user_id).first()
    if not entry_owner_db:
        await query.message.reply_text("Создатель записи не найден.")
        return
    
    inviter_tg = query.from_user
    inviter_db = get_or_create_user(session, inviter_tg.id, inviter_tg.username)

    if inviter_db.id == entry_owner_db.id:
        await query.message.reply_text("Вы не можете пригласить самого себя на РП.")
        return

    context.user_data[f'player_invite_message_id_{entry_id}_{inviter_id}'] = query.message.message_id
    context.user_data[f'player_invite_chat_id_{entry_id}_{inviter_id}'] = query.message.chat_id

    owner_keyboard = InlineKeyboardMarkup([
        [
            InlineKeyboardButton("Согласиться на РП", callback_data=f"player_accept_{inviter_id}_{entry_id}"),
            InlineKeyboardButton("Отказаться от РП", callback_data=f"player_decline_{inviter_id}_{entry_id}")
        ]
    ])
    try:
        await context.bot.send_message(
            chat_id=entry_owner_db.id,
            text=f"Пользователь @{inviter_tg.username or inviter_tg.id} хочет поролить с вами!\n"
                 f"Его сообщение на PLAYERBOARD: {entry.message}\n\n"
                 f"Вы согласны?",
            reply_markup=owner_keyboard
        )
        await query.message.reply_text(f"Приглашение на РП отправлено пользователю @{entry_owner_db.username or entry_owner_db.id}!")
    except TelegramError as e:
        logger.warning(f"Не удалось отправить приглашение на РП пользователю {entry_owner_db.id}: {e}")
        await query.message.reply_text("Не удалось отправить приглашение. Возможно, пользователь заблокировал бота.")

@db_session_for_conversation
@not_banned
async def handle_player_accept_callback(update: Update, context: ContextTypes.DEFAULT_TYPE, session) -> None:
    query = update.callback_query
    await query.answer()

    data = query.data
    parts = data.split('_')
    inviter_id = int(parts[2])
    entry_id = int(parts[3])

    entry_owner_tg = query.from_user
    entry_owner_db = get_or_create_user(session, entry_owner_tg.id, entry_owner_tg.username)

    entry = session.query(PlayerBoardEntry).filter(PlayerBoardEntry.id == entry_id).first()
    if not entry:
        await query.message.reply_text("Эта запись на PLAYERBOARD уже не существует.")
        return

    session.delete(entry)
    
    inviter_db = session.query(User).filter(User.id == inviter_id).first()

    if inviter_db:
        try:
            await context.bot.send_message(
                chat_id=inviter_db.id,
                text=f"Создатель поста @{entry_owner_db.username or entry_owner_db.id} согласился с вами на РП! Обсудите детали в личных сообщениях с @{entry_owner_db.username or entry_owner_db.id}."
            )
        except TelegramError as e:
            logger.warning(f"Не удалось уведомить пригласившего пользователя {inviter_db.id} о согласии: {e}")
    
    try:
        await query.edit_message_text(
            text=f"Вы согласились на РП с @{inviter_db.username or inviter_db.id}. Запись удалена. Обсудите детали в ЛС."
        )
    except TelegramError as e:
        logger.warning(f"Failed to edit owner's message after accepting: {e}")

    context.user_data.pop(f'player_invite_message_id_{entry_id}_{inviter_id}', None)
    context.user_data.pop(f'player_invite_chat_id_{entry_id}_{inviter_id}', None)

@db_session_for_conversation
@not_banned
async def handle_player_decline_callback(update: Update, context: ContextTypes.DEFAULT_TYPE, session) -> None:
    query = update.callback_query
    await query.answer()

    data = query.data
    parts = data.split('_')
    inviter_id = int(parts[2])
    entry_id = int(parts[3])

    entry_owner_tg = query.from_user
    entry_owner_db = get_or_create_user(session, entry_owner_tg.id, entry_owner_tg.username)

    entry = session.query(PlayerBoardEntry).filter(PlayerBoardEntry.id == entry_id).first()
    
    inviter_db = session.query(User).filter(User.id == inviter_id).first()

    if inviter_db:
        try:
            await context.bot.send_message(
                chat_id=inviter_db.id,
                text=f"Создатель поста @{entry_owner_db.username or entry_owner_db.id} отказался от РП с вами. Его запись останется на PLAYERBOARD."
            )
        except TelegramError as e:
            logger.warning(f"Не удалось уведомить пригласившего пользователя {inviter_db.id} об отказе: {e}")

    try:
        await query.edit_message_text(
            text=f"Вы отказались от РП с @{inviter_db.username or inviter_db.id}. Ваша запись останется на PLAYERBOARD."
        )
    except TelegramError as e:
        logger.warning(f"Failed to edit owner's message after declining: {e}")
    
    context.user_data.pop(f'player_invite_message_id_{entry_id}_{inviter_id}', None)
    context.user_data.pop(f'player_invite_chat_id_{entry_id}_{inviter_id}', None)

@db_session
@developer_only
async def add_status(update: Update, context: ContextTypes.DEFAULT_TYPE, session) -> None:
    if len(context.args) < 2:
        await update.message.reply_text("Использование: /addstatus [юзернейм или уникальный код участника] [статус]")
        return

    target_identifier = context.args[0]
    new_status = " ".join(context.args[1:])

    user_db = get_user_by_identifier_db(session, target_identifier)

    if user_db:
        user_db.status_rp = new_status
        await update.message.reply_text(f"Статус РП пользователя @{user_db.username or user_db.id} изменен на '{new_status}'.")
    else:
        await update.message.reply_text(f"Пользователь '{target_identifier}' не найден.")

@db_session
@moderator_or_developer_only
async def add_role_mass(update: Update, context: ContextTypes.DEFAULT_TYPE, session) -> None:
    if len(context.args) < 3:
        help_text = """Упрощенная массовая выдача ролей:

Использование: /add [@username #хэштег НазваниеРоли]

Примеры:
/add @user1 #Виконт Виконт - выдать одну роль одному пользователю
/add @user1 #Виконт Виконт, @user2 #Маг Маг - выдать роли нескольким пользователям

Формат для каждого пользователя: @username #хэштег НазваниеРоли
Разделяйте записи разных пользователей запятой"""
        
        await update.message.reply_text(help_text)
        return
    
    input_text = " ".join(context.args)
    
    user_entries = [entry.strip() for entry in input_text.split(',') if entry.strip()]
    
    results = []
    errors = []
    
    for entry in user_entries:
        parts = entry.split()
        if len(parts) < 3:
            errors.append(f"Неверный формат: {entry}")
            continue
        
        username = parts[0].lstrip('@')
        hashtag = parts[1].lstrip('#')
        role_name = ' '.join(parts[2:])
        
        user_db = get_user_by_identifier_db(session, username)
        if not user_db:
            errors.append(f"Пользователь {username} не найден")
            continue
        
        existing_role = session.query(Role).filter(
            Role.user_id == user_db.id,
            func.lower(Role.hashtag) == func.lower(hashtag)
        ).first()
        
        if existing_role:
            errors.append(f"У пользователя {username} уже есть роль с хэштегом #{hashtag}")
            continue
        
        new_role = Role(
            user=user_db,
            name=role_name,
            hashtag=hashtag
        )
        session.add(new_role)
        
        try:
            await context.bot.send_message(
                chat_id=user_db.id,
                text=f"Администратор @{update.effective_user.username or update.effective_user.id} выдал вам роль: {role_name} (#{hashtag})"
            )
        except TelegramError as e:
            logger.warning(f"Не удалось уведомить пользователя {user_db.id} о выдаче роли: {e}")
        
        results.append(f"Пользователю @{user_db.username or user_db.id} выдана роль {role_name} (#{hashtag})")
    
    report = "Результат выдачи ролей:\n"
    if results:
        report += "\n".join(results) + "\n"
    if errors:
        report += "\nОшибки:\n" + "\n".join(errors)
    
    await update.message.reply_text(report)
    
    session.commit()

@db_session
@moderator_or_developer_only
async def delete_role(update: Update, context: ContextTypes.DEFAULT_TYPE, session) -> None:
    if len(context.args) < 2:
        await update.message.reply_text("Использование: /delete [юзернейм или уникальный код участника] [хештег]")
        return

    target_identifier = context.args[0]
    hashtag = context.args[1].lstrip('#')

    user_db = get_user_by_identifier_db(session, target_identifier)

    if not user_db:
        await update.message.reply_text(f"Пользователь '{target_identifier}' не найден.")
        return

    role_to_delete = session.query(Role).filter(
        Role.user_id == user_db.id,
        func.lower(Role.hashtag) == func.lower(hashtag)
    ).first()

    if role_to_delete:
        role_name = role_to_delete.name
        session.delete(role_to_delete)
        await update.message.reply_text(f"Роль '{role_name}' с хештегом '#{hashtag}' удалена у пользователя @{user_db.username or user_db.id}.")
        try:
            await context.bot.send_message(
                chat_id=user_db.id,
                text=f"Ваша роль {role_name} с хэштегом #{hashtag} была удалена администрацией."
            )
        except TelegramError as e:
            logger.warning(f"Не удалось уведомить пользователя {user_db.id} об удалении роли: {e}")
    else:
        await update.message.reply_text(f"У пользователя @{user_db.username or user_db.id} нет роли с хештегом '#{hashtag}'.")

@db_session
@not_banned
async def check_roles(update: Update, context: ContextTypes.DEFAULT_TYPE, session) -> None:
    user_tg = update.effective_user
    
    target_identifier = None
    if len(context.args) > 0:
        target_identifier = context.args[0]

    if target_identifier:
        user_db = get_user_by_identifier_db(session, target_identifier)
        if not user_db:
            await update.message.reply_text(f"Пользователь '{target_identifier}' не найден.")
            return
    else:
        user_db = get_or_create_user(session, user_tg.id, user_tg.username)

    roles = session.query(Role).filter(Role.user_id == user_db.id).all()

    if roles:
        roles_text = f"Роли пользователя @{user_db.username or user_db.id}:\n\n"
        for i, role in enumerate(roles):
            roles_text += f"{i+1}. {role.name} - #{role.hashtag}\n"
            roles_text += f"   Последняя активность: {role.last_active.strftime('%d.%m.%Y')}\n\n"
        await update.message.reply_text(roles_text)
    else:
        await update.message.reply_text(f"У пользователя @{user_db.username or user_db.id} нет активных ролей.")

async def start_log(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    session = SessionLocal()
    try:
        user_tg = update.effective_user
        user_db = session.query(User).filter(User.id == user_tg.id).first()
        if not user_db or not user_db.is_developer:
            await update.message.reply_text("У вас нет прав для выполнения этой команды.")
            return
        
        global logging_active
        if not logging_active:
            logging_active = True
            save_bot_status()
            await update.message.reply_text("Логирование всех сообщений в файл включено.")
        else:
            await update.message.reply_text("Логирование уже активно.")
    finally:
        session.close()

async def stop_log(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    session = SessionLocal()
    try:
        user_tg = update.effective_user
        user_db = session.query(User).filter(User.id == user_tg.id).first()
        if not user_db or not user_db.is_developer:
            await update.message.reply_text("У вас нет прав для выполнения этой команды.")
            return
        
        global logging_active
        if logging_active:
            logging_active = False
            save_bot_status()
            await update.message.reply_text("Логирование всех сообщений в файл отключено.")
        else:
            await update.message.reply_text("Логирование уже неактивно.")
    finally:
        session.close()

async def file_log(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    session = SessionLocal()
    try:
        user_tg = update.effective_user
        user_db = session.query(User).filter(User.id == user_tg.id).first()
        if not user_db or not user_db.is_developer:
            await update.message.reply_text("У вас нет прав для выполнения этой команды.")
            return
        
        if os.path.exists('log.txt'):
            try:
                with open('log.txt', 'rb') as f:
                    await update.message.reply_document(f)
            except TelegramError as e:
                await update.message.reply_text(f"Ошибка при отправке файла: {e}")
        else:
            await update.message.reply_text("Файл логов не найден.")
    finally:
        session.close()

async def start_filter(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    session = SessionLocal()
    try:
        user_tg = update.effective_user
        user_db = session.query(User).filter(User.id == user_tg.id).first()
        if not user_db or not user_db.is_developer:
            await update.message.reply_text("У вас нет прав для выполнения этой команды.")
            return
        
        global filtering_posts_active
        if not filtering_posts_active:
            filtering_posts_active = True
            save_bot_status()
            await update.message.reply_text("Фильтрация сообщений включена.")
        else:
            await update.message.reply_text("Фильтрация уже активна.")
    finally:
        session.close()

async def stop_filter(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    session = SessionLocal()
    try:
        user_tg = update.effective_user
        user_db = session.query(User).filter(User.id == user_tg.id).first()
        if not user_db or not user_db.is_developer:
            await update.message.reply_text("У вас нет прав для выполнения этой команды.")
            return
        
        global filtering_posts_active
        if filtering_posts_active:
            filtering_posts_active = False
            save_bot_status()
            await update.message.reply_text("Фильтрация сообщений отключена.")
        else:
            await update.message.reply_text("Фильтрация уже неактивна.")
    finally:
        session.close()

async def qyqyqs(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text("Я qyqyqs")

@db_session
@not_banned
async def stata(update: Update, context: ContextTypes.DEFAULT_TYPE, session) -> None:
    user_tg = update.effective_user
    user_db = get_or_create_user(session, user_tg.id, user_tg.username)

    stats = session.query(MessageStat).filter(MessageStat.user_id == user_db.id).first()

    if stats:
        await update.message.reply_text(
            f"Ваша статистика сообщений:\n"
            f"Общее количество сообщений: {stats.message_count}\n"
            f"Количество постов: {stats.post_count}\n"
            f"Последнее обновление: {stats.last_updated.strftime('%d.%m.%Y %H:%M:%S')}"
        )
    else:
        await update.message.reply_text("Статистика по вашим сообщениям пока отсутствует.")

@db_session
@not_banned
async def all_stata(update: Update, context: ContextTypes.DEFAULT_TYPE, session) -> None:
    all_stats = session.query(MessageStat).order_by(MessageStat.message_count.desc()).all()

    if all_stats:
        stats_text = "Общая статистика сообщений (топ 20):\n\n"
        for i, stat in enumerate(all_stats[:20]):
            user = session.query(User).filter(User.id == stat.user_id).first()
            username = user.username if user else f"ID:{stat.user_id}"
            stats_text += f"{i+1}. @{username}: {stat.message_count} сообщений, {stat.post_count} постов\n"
        
        if len(all_stats) > 20:
            stats_text += "\n..."

        await update.message.reply_text(stats_text)
    else:
        await update.message.reply_text("Общая статистика сообщений пока отсутствует.")

@db_session
@not_banned
async def links(update: Update, context: ContextTypes.DEFAULT_TYPE, session) -> None:
    links_message = """Инфоканал - @MultiverseRp
Переходник - @AdapterMultiverseRp
Тикток - https://www.tiktok.com/@multiverserpproject?_r=1&_t=ZM-92Dc1DKOHm7
Владелец - @Mr011022011"""
    message_source = update.callback_query.message if update.callback_query else update.message
    await message_source.reply_text(links_message, disable_web_page_preview=True)
    if update.callback_query:
        await update.callback_query.answer()

@db_session
@developer_only
async def ban_user(update: Update, context: ContextTypes.DEFAULT_TYPE, session) -> None:
    if len(context.args) < 1:
        await update.message.reply_text("Использование: /ban [юзернейм или уникальный код участника]")
        return

    target_identifier = context.args[0]
    user_db = get_user_by_identifier_db(session, target_identifier)

    if not user_db:
        await update.message.reply_text(f"Пользователь '{target_identifier}' не найден.")
        return

    if user_db.is_banned:
        await update.message.reply_text(f"Пользователь @{user_db.username or user_db.id} уже забанен.")
        return
    
    if user_db.is_developer:
        await update.message.reply_text(f"Невозможно забанить разработчика @{user_db.username or user_db.id}.")
        return

    user_db.is_banned = True
    await update.message.reply_text(f"Пользователь @{user_db.username or user_db.id} забанен.")
    try:
        await context.bot.send_message(
            chat_id=user_db.id,
            text="Вы были забанены администрацией и не можете использовать бота."
        )
    except TelegramError as e:
        logger.warning(f"Не удалось уведомить пользователя {user_db.id} о бане: {e}")

@db_session
@developer_only
async def unban_user(update: Update, context: ContextTypes.DEFAULT_TYPE, session) -> None:
    if len(context.args) < 1:
        await update.message.reply_text("Использование: /unban [юзернейм или уникальный код участника]")
        return

    target_identifier = context.args[0]
    user_db = get_user_by_identifier_db(session, target_identifier)

    if not user_db:
        await update.message.reply_text(f"Пользователь '{target_identifier}' не найден.")
        return

    if not user_db.is_banned:
        await update.message.reply_text(f"Пользователь @{user_db.username or user_db.id} не забанен.")
        return

    user_db.is_banned = False
    await update.message.reply_text(f"Пользователь @{user_db.username or user_db.id} разбанен.")
    try:
        await context.bot.send_message(
            chat_id=user_db.id,
            text="Вы были разбанены администрацией и теперь можете использовать бота."
        )
    except TelegramError as e:
        logger.warning(f"Не удалось уведомить пользователя {user_db.id} о разбане: {e}")

@db_session
@not_banned
async def send_money(update: Update, context: ContextTypes.DEFAULT_TYPE, session) -> None:
    if update.message.reply_to_message:
        reply_to = update.message.reply_to_message
        if reply_to.from_user:
            target_identifier = str(reply_to.from_user.id)
        else:
            await update.message.reply_text("Не удалось определить пользователя, которому нужно отправить средства.")
            return
        
        if len(context.args) < 2:
            await update.message.reply_text("Использование (в ответ на сообщение): /send [сумма] [ON/OP]")
            return
        
        amount_str = context.args[0]
        currency = context.args[1].upper()
    else:
        if len(context.args) < 3:
            await update.message.reply_text("Использование: /send [получатель] [сумма] [ON/OP]")
            return
        
        target_identifier = context.args[0]
        amount_str = context.args[1]
        currency = context.args[2].upper()
    
    if currency not in ["ON", "OP"]:
        await update.message.reply_text("Валюта должна быть ON или OP.")
        return
    
    try:
        amount = int(amount_str)
        if amount <= 0:
            await update.message.reply_text("Сумма должна быть положительным числом.")
            return
    except ValueError:
        await update.message.reply_text("Сумма должна быть числом.")
        return
    
    sender_tg = update.effective_user
    sender_db = get_or_create_user(session, sender_tg.id, sender_tg.username)
    
    recipient_db = get_user_by_identifier_db(session, target_identifier)
    if not recipient_db:
        await update.message.reply_text(f"Получатель '{target_identifier}' не найден.")
        return
    
    if sender_db.id == recipient_db.id:
        await update.message.reply_text("Нельзя отправить средства самому себе.")
        return
    
    if currency == "ON":
        if sender_db.on_balance < amount:
            await update.message.reply_text(f"Недостаточно ОН. Ваш баланс: {sender_db.on_balance} ОН")
            return
        sender_db.on_balance -= amount
        recipient_db.on_balance += amount
    else:
        if sender_db.op_balance < amount:
            await update.message.reply_text(f"Недостаточно ОП. Ваш баланс: {sender_db.op_balance} ОП")
            return
        sender_db.op_balance -= amount
        recipient_db.op_balance += amount
    
    try:
        await context.bot.send_message(
            chat_id=recipient_db.id,
            text=f"Вы получили перевод от @{sender_tg.username or sender_tg.id}!\n"
                 f"Сумма: {amount} {currency}\n"
                 f"Ваш баланс {currency}: {recipient_db.on_balance if currency == 'ON' else recipient_db.op_balance}"
        )
    except TelegramError as e:
        logger.warning(f"Не удалось уведомить получателя {recipient_db.id} о переводе: {e}")
    
    await update.message.reply_text(
        f"Перевод выполнен успешно!\n"
        f"Получатель: @{recipient_db.username or recipient_db.id}\n"
        f"Сумма: {amount} {currency}\n"
        f"Ваш баланс {currency}: {sender_db.on_balance if currency == 'ON' else sender_db.op_balance}"
    )

@db_session_for_conversation
@not_banned
async def start_create_check(update: Update, context: ContextTypes.DEFAULT_TYPE, session) -> int:
    user_tg = update.effective_user
    user_db = get_or_create_user(session, user_tg.id, user_tg.username)
    
    await update.message.reply_text(
        "Начинаем создание чека. Введите сумму чека:"
    )
    
    context.user_data['check_creator_id'] = user_db.id
    return STATE_CREATE_CHECK_AMOUNT

@db_session_for_conversation
async def create_check_amount(update: Update, context: ContextTypes.DEFAULT_TYPE, session) -> int:
    try:
        amount = int(update.message.text.strip())
        if amount <= 0:
            await update.message.reply_text("Сумма должна быть положительным числом. Попробуйте еще раз:")
            return STATE_CREATE_CHECK_AMOUNT
    except ValueError:
        await update.message.reply_text("Пожалуйста, введите число. Попробуйте еще раз:")
        return STATE_CREATE_CHECK_AMOUNT
    
    context.user_data['check_amount'] = amount
    
    await update.message.reply_text(
        "Теперь введите валюту чека (ON или OP):"
    )
    return STATE_CREATE_CHECK_CURRENCY

@db_session_for_conversation
async def create_check_currency(update: Update, context: ContextTypes.DEFAULT_TYPE, session) -> int:
    currency = update.message.text.strip().upper()
    if currency not in ["ON", "OP"]:
        await update.message.reply_text("Валюта должна быть ON или OP. Попробуйте еще раз:")
        return STATE_CREATE_CHECK_CURRENCY
    
    creator_id = context.user_data.get('check_creator_id')
    if not creator_id:
        await update.message.reply_text("Ошибка: не найден создатель чека. Начните заново.")
        return ConversationHandler.END
    
    creator = session.query(User).filter(User.id == creator_id).first()
    if not creator:
        await update.message.reply_text("Ошибка: создатель не найден. Начните заново.")
        return ConversationHandler.END
    
    amount = context.user_data['check_amount']
    if currency == "ON":
        if creator.on_balance < amount:
            await update.message.reply_text(
                f"У вас недостаточно ОН для создания чека.\n"
                f"Ваш баланс: {creator.on_balance} ОН\n"
                f"Требуется: {amount} ОН\n"
                f"Введите другую сумму или отмените создание чека."
            )
            return STATE_CREATE_CHECK_AMOUNT
    else:
        if creator.op_balance < amount:
            await update.message.reply_text(
                f"У вас недостаточно ОП для создания чека.\n"
                f"Ваш баланс: {creator.op_balance} ОП\n"
                f"Требуется: {amount} ОП\n"
                f"Введите другую сумму или отмените создание чека."
            )
            return STATE_CREATE_CHECK_AMOUNT
    
    context.user_data['check_currency'] = currency
    
    await update.message.reply_text(
        "Теперь введите максимальное количество использований чека (или 0 для неограниченного использования):"
    )
    return STATE_CREATE_CHECK_MAX_USES

@db_session_for_conversation
async def create_check_max_uses(update: Update, context: ContextTypes.DEFAULT_TYPE, session) -> int:
    try:
        max_uses = int(update.message.text.strip())
        if max_uses < 0:
            await update.message.reply_text("Количество использований должно быть неотрицательным числом. Попробуйте еще раз:")
            return STATE_CREATE_CHECK_MAX_USES
    except ValueError:
        await update.message.reply_text("Пожалуйста, введите число. Попробуйте еще раз:")
        return STATE_CREATE_CHECK_MAX_USES
    
    if max_uses == 0:
        max_uses = 999999
    
    context.user_data['check_max_uses'] = max_uses
    
    await update.message.reply_text(
        "Теперь введите пароль для чека (или 'нет' если пароль не нужен):"
    )
    return STATE_CREATE_CHECK_PASSWORD

@db_session_for_conversation
async def create_check_password(update: Update, context: ContextTypes.DEFAULT_TYPE, session) -> int:
    password = update.message.text.strip()
    if password.lower() == 'нет':
        password = None
    
    context.user_data['check_password'] = password
    
    await update.message.reply_text(
        "Теперь введите описание чека (или 'нет' если описание не нужно):"
    )
    return STATE_CREATE_CHECK_DESCRIPTION

@db_session_for_conversation
async def create_check_description(update: Update, context: ContextTypes.DEFAULT_TYPE, session) -> int:
    description = update.message.text.strip()
    if description.lower() == 'нет':
        description = None
    
    creator_id = context.user_data.get('check_creator_id')
    if not creator_id:
        await update.message.reply_text("Ошибка: не найден создатель чека. Начните заново.")
        return ConversationHandler.END
    
    creator = session.query(User).filter(User.id == creator_id).first()
    if not creator:
        await update.message.reply_text("Ошибка: создатель не найден. Начните заново.")
        return ConversationHandler.END
    
    amount = context.user_data['check_amount']
    currency = context.user_data['check_currency']
    
    if currency == "ON":
        creator.on_balance -= amount
    else:
        creator.op_balance -= amount
    
    new_check = Check(
        creator=creator,
        amount=amount,
        currency=currency,
        description=description,
        max_uses=context.user_data['check_max_uses'],
        password=context.user_data['check_password']
    )
    session.add(new_check)
    session.flush()
    
    max_uses_display = "Неограниченно" if context.user_data['check_max_uses'] == 999999 else context.user_data['check_max_uses']
    
    await update.message.reply_text(
        f"Чек успешно создан!\n"
        f"Код чека: {new_check.unique_code}\n"
        f"Сумма: {amount} {currency}\n"
        f"Макс. использований: {max_uses_display}\n"
        f"{'Пароль: ' + context.user_data['check_password'] if context.user_data['check_password'] else 'Без пароля'}\n"
        f"{'Описание: ' + description if description else ''}\n"
        f"Ваш баланс {currency}: {creator.on_balance if currency == 'ON' else creator.op_balance}"
    )
    
    for key in ['check_creator_id', 'check_amount', 'check_currency', 'check_max_uses', 'check_password']:
        context.user_data.pop(key, None)
    
    return ConversationHandler.END

@db_session
@not_banned
async def take_check(update: Update, context: ContextTypes.DEFAULT_TYPE, session) -> None:
    if len(context.args) < 1:
        await update.message.reply_text("Использование: /takecheck [код_чека]")
        return
    
    check_code = context.args[0]
    
    user_tg = update.effective_user
    user_db = get_or_create_user(session, user_tg.id, user_tg.username)
    
    check = session.query(Check).filter(Check.unique_code == check_code).first()
    
    if not check:
        await update.message.reply_text("Чек с таким кодом не найден.")
        return
    
    if not check.is_active:
        await update.message.reply_text("Этот чек больше не активен.")
        return
    
    if check.password:
        if len(context.args) < 2:
            await update.message.reply_text("Для этого чека требуется пароль. Использование: /takecheck [код_чека] [пароль]")
            return
        
        password = context.args[1]
        if password != check.password:
            await update.message.reply_text("Неверный пароль.")
            return
    
    if user_db.id == check.creator_id:
        await update.message.reply_text("Вы не можете активировать свой собственный чек.")
        return
    
    check.current_uses += 1
    
    if check.currency == "ON":
        user_db.on_balance += check.amount
    else:
        user_db.op_balance += check.amount
    
    await update.message.reply_text(
        f"Чек успешно активирован!\n"
        f"Вы получили: {check.amount} {check.currency}\n"
        f"Ваш баланс {check.currency}: {user_db.on_balance if check.currency == 'ON' else user_db.op_balance}"
    )
    
    try:
        creator = session.query(User).filter(User.id == check.creator_id).first()
        if creator:
            await context.bot.send_message(
                chat_id=creator.id,
                text=f"Ваш чек (код: {check.unique_code}) был активирован пользователем @{user_tg.username or user_tg.id}."
            )
    except TelegramError as e:
        logger.warning(f"Не удалось уведомить создателя чека {check.creator_id}: {e}")

@db_session
@not_banned
async def list_checks(update: Update, context: ContextTypes.DEFAULT_TYPE, session) -> None:
    user_tg = update.effective_user
    user_db = get_or_create_user(session, user_tg.id, user_tg.username)
    
    checks = session.query(Check).filter(Check.creator_id == user_db.id, Check.current_uses < Check.max_uses).all()
    
    if not checks:
        await update.message.reply_text("У вас нет активных чеков.")
        return
    
    response = f"Ваши активные чеки ({len(checks)} шт.):\n\n"
    
    for i, check in enumerate(checks):
        max_uses_display = "∞" if check.max_uses == 999999 else check.max_uses
        
        response += f"{i+1}. Код: {check.unique_code}\n"
        response += f"   Сумма: {check.amount} {check.currency}\n"
        response += f"   Использовано: {check.current_uses}/{max_uses_display}\n"
        if check.password:
            response += f"   Пароль: {check.password}\n"
        if check.description:
            response += f"   Описание: {check.description[:50]}...\n"
        response += f"   Создан: {check.created_at.strftime('%d.%m.%Y %H:%M')}\n\n"
    
    await update.message.reply_text(response)

@db_session
@not_banned
async def delete_check(update: Update, context: ContextTypes.DEFAULT_TYPE, session) -> None:
    if len(context.args) < 1:
        await update.message.reply_text("Использование: /deletecheck [код_чека]")
        return
    
    check_code = context.args[0]
    
    user_tg = update.effective_user
    user_db = get_or_create_user(session, user_tg.id, user_tg.username)
    
    check = session.query(Check).filter(Check.unique_code == check_code).first()
    
    if not check:
        await update.message.reply_text("Чек с таким кодом не найден.")
        return
    
    if user_db.id != check.creator_id and not user_db.is_developer:
        await update.message.reply_text("Вы можете удалять только свои чеки.")
        return
    
    if check.current_uses == 0:
        if check.currency == "ON":
            user_db.on_balance += check.amount
        else:
            user_db.op_balance += check.amount
    
    session.delete(check)
    
    await update.message.reply_text(
        f"Чек успешно удален.\n"
        f"{'Средства возвращены на ваш баланс.' if check.current_uses == 0 else ''}"
    )

@db_session
async def button_handler(update: Update, context: ContextTypes.DEFAULT_TYPE, session) -> None:
    query = update.callback_query
    
    if not query:
        logger.error("button_handler called without a callback_query.")
        return

    user_tg = query.from_user
    
    user_db = session.query(User).filter(User.id == user_tg.id).first()
    if user_db and user_db.is_banned:
        await query.answer("Вы забанены и не можете использовать бота.")
        return
    
    if query.message.chat.type != "private" and query.message.chat.id not in ALLOWED_CHAT_IDS:
         await query.answer("Бот не работает в этом чате. Используйте его в разрешенных группах или в личных сообщениях.")
         return

    data = query.data
    
    if data == "start":
        await start(update, context)
    elif data == "help":
        await help_command(update, context)
    elif data == "profile":
        await profile(update, context)
    elif data == "playerboard_list":
        await playerboard_list(update, context)
    elif data == "links":
        await links(update, context)
    elif data == "newbie_info":
        await newbie_info(update, context)
    elif data.startswith("show_user_nagrads_"):
        await show_my_nagrads(update, context)
    elif data == "send_anketa_callback":
        await send_anketa_callback(update, context)
    elif data == "support_dialog":
        await start_support_dialog(update, context)
    elif data == "playerboard_create":
        await start_player_dialog(update, context)
    elif data == "info_command":
        await info_command(update, context)
    elif data == "none":
        await query.answer("Это действие уже выполнено или неактивно.")
    else:
        await query.answer(f"Неизвестное действие: {data}")

async def handle_unknown_private_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if update.effective_chat.type == "private":
        if update.message and update.message.text and not update.message.text.startswith('/'):
            await update.message.reply_text(
                "Извините, я не понял вашу команду или сообщение. Пожалуйста, используйте кнопки меню или /help."
            )
        elif update.callback_query:
            try:
                await update.callback_query.answer("Неизвестное действие. Пожалуйста, используйте кнопки меню или /help.")
            except TelegramError:
                pass

async def post_init(application: Application) -> None:
    try:
        await application.bot.send_message(chat_id=-1003431402721, text="БОТ СЕЙЧАС ВКЛЮЧЕН!")
    except Exception as e:
        logger.error(f"Не удалось отправить сообщение о включении бота: {e}")

async def post_shutdown(application: Application) -> None:
    try:
        await application.bot.send_message(chat_id=-1003431402721, text="БОТ СЕЙЧАС ВЫКЛЮЧАЕТСЯ!")
    except Exception as e:
        logger.error(f"Не удалось отправить сообщение о выключении бота: {e}")

def main() -> None:
    create_tables()
    load_bot_status()

    if TOKEN == "YOUR_ACTUAL_BOT_TOKEN_HERE":
        logger.critical("ОШИБКА: Токен бота не был заменен! Работа бота приостановлена.")
        return

    application = Application.builder().token(TOKEN).post_init(post_init).post_shutdown(post_shutdown).build()

    job_queue = application.job_queue
    job_queue.run_repeating(check_inactive_roles_with_warnings,
                            interval=datetime.timedelta(days=1),
                            first=datetime.time(hour=3, minute=0),
                            name="check_inactive_roles_with_warnings_job")

    class ChatFilter(filters.BaseFilter):
        def __init__(self, allowed_chat_ids: list[int]):
            super().__init__()
            self.allowed_chat_ids = allowed_chat_ids

        def filter(self, update: Update):
            if update.effective_chat:
                if update.effective_chat.type == "private":
                    return True
                return update.effective_chat.id in self.allowed_chat_ids
            return False

    allowed_chats_filter = ChatFilter(ALLOWED_CHAT_IDS)

    anketa_conversation = ConversationHandler(
        entry_points=[
            CommandHandler("sendanketa", send_anketa_start, filters=allowed_chats_filter),
            CallbackQueryHandler(send_anketa_callback, pattern="^send_anketa_callback$"),
        ],
        states={
            STATE_ANKETA_MESSAGE: [
                MessageHandler(
                    (filters.TEXT & ~filters.COMMAND) | 
                    filters.PHOTO | 
                    filters.VIDEO | 
                    filters.ANIMATION | 
                    filters.Document.ALL & allowed_chats_filter, 
                    anketa_message
                )
            ],
            STATE_ANKETA_CLARIFY: [
                MessageHandler(
                    (filters.TEXT & ~filters.COMMAND) | 
                    filters.PHOTO | 
                    filters.VIDEO | 
                    filters.Document.ALL & allowed_chats_filter, 
                    clarify_message
                )
            ],
        },
        fallbacks=[
            CommandHandler("done_anketa", done_anketa_dialog, filters=allowed_chats_filter),
            CommandHandler("done_clarify", done_clarify_dialog, filters=allowed_chats_filter),
            CommandHandler("start", start, filters=allowed_chats_filter),
            CallbackQueryHandler(start, pattern="^start$")
        ],
        allow_reentry=True
    )
    application.add_handler(anketa_conversation)

    add_nagrad_conversation = ConversationHandler(
        entry_points=[
            CommandHandler("addnagrad", start_add_nagrad, filters=allowed_chats_filter),
        ],
        states={
            STATE_ADD_NAGRAD_NAME: [MessageHandler(filters.TEXT & ~filters.COMMAND & allowed_chats_filter, add_nagrad_name)],
            STATE_ADD_NAGRAD_DESCRIPTION: [MessageHandler(filters.TEXT & ~filters.COMMAND & allowed_chats_filter, add_nagrad_description)],
            STATE_ADD_NAGRAD_PHOTO: [MessageHandler(filters.PHOTO | filters.TEXT & ~filters.COMMAND & allowed_chats_filter, add_nagrad_photo)],
            STATE_ADD_NAGRAD_COST: [MessageHandler(filters.TEXT & ~filters.COMMAND & allowed_chats_filter, add_nagrad_cost)],
            STATE_ADD_NAGRAD_TARGET_USER: [MessageHandler(filters.TEXT & ~filters.COMMAND & allowed_chats_filter, add_nagrad_target_user)],
        },
        fallbacks=[
            CommandHandler("start", start, filters=allowed_chats_filter),
            CallbackQueryHandler(start, pattern="^start$")
        ],
        allow_reentry=True
    )
    application.add_handler(add_nagrad_conversation)

    create_check_conversation = ConversationHandler(
        entry_points=[
            CommandHandler("createcheck", start_create_check, filters=allowed_chats_filter),
        ],
        states={
            STATE_CREATE_CHECK_AMOUNT: [MessageHandler(filters.TEXT & ~filters.COMMAND & allowed_chats_filter, create_check_amount)],
            STATE_CREATE_CHECK_CURRENCY: [MessageHandler(filters.TEXT & ~filters.COMMAND & allowed_chats_filter, create_check_currency)],
            STATE_CREATE_CHECK_MAX_USES: [MessageHandler(filters.TEXT & ~filters.COMMAND & allowed_chats_filter, create_check_max_uses)],
            STATE_CREATE_CHECK_PASSWORD: [MessageHandler(filters.TEXT & ~filters.COMMAND & allowed_chats_filter, create_check_password)],
            STATE_CREATE_CHECK_DESCRIPTION: [MessageHandler(filters.TEXT & ~filters.COMMAND & allowed_chats_filter, create_check_description)],
        },
        fallbacks=[
            CommandHandler("start", start, filters=allowed_chats_filter),
            CallbackQueryHandler(start, pattern="^start$")
        ],
        allow_reentry=True
    )
    application.add_handler(create_check_conversation)

    support_conversation = ConversationHandler(
        entry_points=[
            CommandHandler("support", start_support_dialog, filters=allowed_chats_filter),
            CallbackQueryHandler(start_support_dialog, pattern="^support_dialog$"),
            CallbackQueryHandler(start_reply_to_support, pattern="^support_reply_"),
        ],
        states={
            STATE_SUPPORT_MESSAGE: [
                MessageHandler(
                    (filters.TEXT & ~filters.COMMAND) | 
                    filters.PHOTO | 
                    filters.VIDEO | 
                    filters.ANIMATION | 
                    filters.Document.ALL & allowed_chats_filter, 
                    support_message
                )
            ],
            STATE_SUPPORT_REPLY: [
                MessageHandler(
                    (filters.TEXT & ~filters.COMMAND) & allowed_chats_filter, 
                    support_reply_message
                )
            ],
        },
        fallbacks=[
            CommandHandler("done_support", done_support_dialog, filters=allowed_chats_filter),
            CommandHandler("start", start, filters=allowed_chats_filter),
            CallbackQueryHandler(start, pattern="^start$")
        ],
        allow_reentry=True
    )
    application.add_handler(support_conversation)

    playerboard_conversation = ConversationHandler(
        entry_points=[
            CommandHandler("playerboard", playerboard_list, filters=allowed_chats_filter),
            CallbackQueryHandler(playerboard_list, pattern="^playerboard_list$"),
            CallbackQueryHandler(start_player_dialog, pattern="^playerboard_create$"),
        ],
        states={
            STATE_PLAYERBOARD_MESSAGE: [MessageHandler(filters.TEXT & ~filters.COMMAND & allowed_chats_filter, player_message_step)],
            STATE_PLAYERBOARD_ROLES: [MessageHandler(filters.TEXT & ~filters.COMMAND & allowed_chats_filter, player_roles_step)],
        },
        fallbacks=[
            CommandHandler("start", start, filters=allowed_chats_filter),
            CallbackQueryHandler(start, pattern="^start$")
        ],
        allow_reentry=True
    )
    application.add_handler(playerboard_conversation)

    send_info_conversation = ConversationHandler(
        entry_points=[
            CommandHandler("SendInfo", start_send_info, filters=allowed_chats_filter),
        ],
        states={
            STATE_SEND_INFO_CONTENT: [
                MessageHandler(
                    (filters.TEXT & ~filters.COMMAND) | 
                    filters.PHOTO | 
                    filters.VIDEO | 
                    filters.ANIMATION & allowed_chats_filter, 
                    send_info_content
                )
            ],
        },
        fallbacks=[
            CommandHandler("Done_info", done_info_dialog, filters=allowed_chats_filter),
            CommandHandler("start", start, filters=allowed_chats_filter),
            CallbackQueryHandler(start, pattern="^start$")
        ],
        allow_reentry=True
    )
    application.add_handler(send_info_conversation)

    application.add_handler(CommandHandler("start", start, filters=allowed_chats_filter))
    application.add_handler(CommandHandler("help", help_command, filters=allowed_chats_filter))
    application.add_handler(CommandHandler("profile", profile, filters=allowed_chats_filter))
    application.add_handler(CommandHandler("send", send_money, filters=allowed_chats_filter))
    application.add_handler(CommandHandler("addstatus", add_status, filters=allowed_chats_filter))
    application.add_handler(CommandHandler("check", check_roles, filters=allowed_chats_filter))
    
    application.add_handler(CommandHandler("Info", info_command, filters=allowed_chats_filter))
    application.add_handler(CommandHandler("InfoON", info_on, filters=allowed_chats_filter))
    application.add_handler(CommandHandler("InfoOFF", info_off, filters=allowed_chats_filter))
    application.add_handler(CommandHandler("deleteinfopost", delete_info_post, filters=allowed_chats_filter))
    
    application.add_handler(CommandHandler("add", add_role_mass, filters=allowed_chats_filter))
    
    application.add_handler(CommandHandler("delete", delete_role, filters=allowed_chats_filter))
    
    application.add_handler(CommandHandler("deleteplayerboard", delete_playerboard_entry, filters=allowed_chats_filter))
    application.add_handler(CommandHandler("checkpost", check_post_stats, filters=allowed_chats_filter))
    application.add_handler(CommandHandler("CheckRole", check_role, filters=allowed_chats_filter))
    application.add_handler(CommandHandler("reset", reset_user, filters=allowed_chats_filter))
    
    application.add_handler(CommandHandler("QyqyqysON", qyqyqys_on, filters=allowed_chats_filter))
    application.add_handler(CommandHandler("QyqyqysOP", qyqyqys_op, filters=allowed_chats_filter))
    
    application.add_handler(CommandHandler("checknagrad", check_nagrad, filters=allowed_chats_filter))
    application.add_handler(CommandHandler("sellnagrad", sell_nagrad, filters=allowed_chats_filter))
    application.add_handler(CommandHandler("Nagrada_On", nagrada_on, filters=allowed_chats_filter))
    application.add_handler(CommandHandler("Nagrada_Off", nagrada_off, filters=allowed_chats_filter))
    application.add_handler(CommandHandler("deletenagrad", delete_my_nagrad, filters=allowed_chats_filter))
    application.add_handler(CommandHandler("SendNagrada", send_nagrada, filters=allowed_chats_filter))
    application.add_handler(CommandHandler("Nagrada", get_nagrada_details, filters=allowed_chats_filter))
    
    application.add_handler(CommandHandler("takecheck", take_check, filters=allowed_chats_filter))
    application.add_handler(CommandHandler("listchecks", list_checks, filters=allowed_chats_filter))
    application.add_handler(CommandHandler("deletecheck", delete_check, filters=allowed_chats_filter))
    
    application.add_handler(CommandHandler("stata", stata, filters=allowed_chats_filter))
    application.add_handler(CommandHandler("allstata", all_stata, filters=allowed_chats_filter))
    application.add_handler(CommandHandler("links", links, filters=allowed_chats_filter))
    application.add_handler(CommandHandler("ban", ban_user, filters=allowed_chats_filter))
    application.add_handler(CommandHandler("unban", unban_user, filters=allowed_chats_filter))

    application.add_handler(CommandHandler("startlog", start_log, filters=allowed_chats_filter))
    application.add_handler(CommandHandler("stoplog", stop_log, filters=allowed_chats_filter))
    application.add_handler(CommandHandler("filelog", file_log, filters=allowed_chats_filter))
    application.add_handler(CommandHandler("startfilter", start_filter, filters=allowed_chats_filter))
    application.add_handler(CommandHandler("stopfilter", stop_filter, filters=allowed_chats_filter))
    application.add_handler(CommandHandler("qyqyqs", qyqyqs, filters=allowed_chats_filter))

    application.add_handler(MessageHandler(
        filters.Regex(r"^/player_contact_(\d+)$") & allowed_chats_filter,
        player_contact
    ))

    application.add_handler(CallbackQueryHandler(handle_anketa_callback, pattern="^anketa_"))
    application.add_handler(CallbackQueryHandler(handle_support_callback, pattern="^support_end_dialog_"))
    application.add_handler(CallbackQueryHandler(handle_player_invite_callback, pattern="^player_invite_"))
    application.add_handler(CallbackQueryHandler(handle_player_accept_callback, pattern="^player_accept_"))
    application.add_handler(CallbackQueryHandler(handle_player_decline_callback, pattern="^player_decline_"))

    application.add_handler(CallbackQueryHandler(
        button_handler,
        pattern="^(?!support_reply_|support_end_dialog_|player_invite_|player_accept_|player_decline_|anketa_).*$"
    ))

    application.add_handler(MessageHandler(
        (filters.TEXT | filters.PHOTO | filters.VIDEO | filters.ANIMATION | filters.Document.ALL) & 
        filters.ChatType.GROUPS & allowed_chats_filter, 
        log_and_stats_message_handler, 
        block=False
    ))
    
    application.add_handler(MessageHandler(filters.ALL & filters.ChatType.PRIVATE, handle_unknown_private_message))
    application.add_handler(CallbackQueryHandler(handle_unknown_private_message))

    logger.info("Бот запущен...")
    application.run_polling(allowed_updates=Update.ALL_TYPES)
    logger.info("Бот остановлен.")
    
    save_bot_status()

from flask import Flask, jsonify
import threading
import time

app = Flask(__name__)

@app.route('/')
def home():
    return jsonify({"status": "bot_is_running", "project": "Multiverse-RP"})

@app.route('/health')
def health():
    return jsonify({"status": "healthy"})

@app.route('/ping')
def ping():
    return "pong"

def run_web_server():
    port = int(os.getenv('PORT', 8080))
    app.run(host='0.0.0.0', port=port, debug=False)

def main_with_web():
    logger.info("Запуск веб-сервера для Render...")
    
    web_thread = threading.Thread(target=run_web_server, daemon=True)
    web_thread.start()
    
    time.sleep(2)
    
    main()

if __name__ == "__main__":
    main_with_web()