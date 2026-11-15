"""Tiny auth layer: users + sessions in SQLite (standard library only).
Passwords are salted and hashed (PBKDF2-SHA256), never stored in plain text.
Sessions are random tokens; only their SHA-256 hash is stored in the database."""
import hashlib, hmac, os, re, secrets, sqlite3, time
from contextlib import closing

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(ROOT, "instance", "users.db")
ITERATIONS = 200_000
EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")

_fails = {}  # simple login rate limit: key -> [count, locked_until]


def _conn():
    c = sqlite3.connect(DB_PATH)
    c.row_factory = sqlite3.Row
    return c


def init():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    with closing(_conn()) as c, c:
        c.executescript("""
        CREATE TABLE IF NOT EXISTS users(
            id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT NOT NULL, email TEXT NOT NULL UNIQUE,
            salt BLOB NOT NULL, pw_hash BLOB NOT NULL, created REAL NOT NULL);
        CREATE TABLE IF NOT EXISTS sessions(
            token_hash TEXT PRIMARY KEY, user_id INTEGER NOT NULL, expires REAL NOT NULL);
        """)


def _hash(password, salt):
    return hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, ITERATIONS)


def clean_email(email):
    return (email or "").strip().lower()


def create_user(name, email, password):
    name, email = (name or "").strip(), clean_email(email)
    if not 2 <= len(name) <= 60:
        raise ValueError("Please enter your name (2 to 60 characters).")
    if not EMAIL_RE.match(email) or len(email) > 120:
        raise ValueError("Please enter a valid email address.")
    if len(password or "") < 8 or len(password) > 128:
        raise ValueError("Password must be at least 8 characters.")
    salt = secrets.token_bytes(16)
    try:
        with closing(_conn()) as c, c:
            cur = c.execute("INSERT INTO users(name,email,salt,pw_hash,created) VALUES(?,?,?,?,?)",
                            (name, email, salt, _hash(password, salt), time.time()))
            return {"id": cur.lastrowid, "name": name, "email": email}
    except sqlite3.IntegrityError:
        raise ValueError("An account with this email already exists. Please log in.")


def verify_user(email, password):
    email = clean_email(email)
    with closing(_conn()) as c:
        row = c.execute("SELECT * FROM users WHERE email=?", (email,)).fetchone()
    salt = row["salt"] if row else b"\0" * 16  # same work whether or not the user exists
    ok = hmac.compare_digest(_hash(password or "", salt), row["pw_hash"] if row else b"\0" * 32)
    return {"id": row["id"], "name": row["name"], "email": row["email"]} if (row and ok) else None


# ---- rate limit (5 wrong attempts -> 60 s lock)
def locked(key):
    f = _fails.get(key)
    return bool(f and f[1] > time.time())


def record_fail(key):
    f = _fails.setdefault(key, [0, 0.0])
    f[0] += 1
    if f[0] >= 5:
        f[0], f[1] = 0, time.time() + 60


def clear_fails(key):
    _fails.pop(key, None)


# ---- sessions
def _th(token):
    return hashlib.sha256(token.encode()).hexdigest()


def create_session(user_id, days):
    token = secrets.token_urlsafe(32)
    with closing(_conn()) as c, c:
        c.execute("DELETE FROM sessions WHERE expires < ?", (time.time(),))
        c.execute("INSERT INTO sessions(token_hash,user_id,expires) VALUES(?,?,?)",
                  (_th(token), user_id, time.time() + days * 86400))
    return token


def user_for_token(token):
    if not token:
        return None
    with closing(_conn()) as c:
        row = c.execute("""SELECT u.id, u.name, u.email FROM sessions s JOIN users u ON u.id = s.user_id
                           WHERE s.token_hash=? AND s.expires > ?""", (_th(token), time.time())).fetchone()
    return dict(row) if row else None


def delete_session(token):
    if token:
        with closing(_conn()) as c, c:
            c.execute("DELETE FROM sessions WHERE token_hash=?", (_th(token),))