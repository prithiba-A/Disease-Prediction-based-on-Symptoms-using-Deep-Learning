import streamlit as st
import sqlite3
import bcrypt

# ✅ Function to get SQLite DB connection
def get_db_connection():
    try:
        conn = sqlite3.connect("database.db")  
        cursor = conn.cursor()
        
        # ✅ Create users table if not exists
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                email TEXT UNIQUE NOT NULL,
                password TEXT NOT NULL
            )
        """)
        
        conn.commit()
        return conn
    except sqlite3.Error as e:
        st.error(f"❌ Database Error: {e}")  # ✅ Debugging error messages
        return None

# ✅ Function to hash passwords securely
def hash_password(password):
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()

# ✅ Function to verify passwords
def verify_password(password, hashed):
    return bcrypt.checkpw(password.encode(), hashed.encode())

# ✅ Sign-up function (Prevents auto-login after sign-up)
def signup(username, email, password):
    conn = get_db_connection()
    if not conn:
        return False, "❌ Database connection failed."

    c = conn.cursor()
    hashed_password = hash_password(password)  # Hash the password before storing

    try:
        c.execute("INSERT INTO users (username, email, password) VALUES (?, ?, ?)", 
                  (username, email, hashed_password))
        conn.commit()
        return True, "✅ Account created successfully! Please log in."
    except sqlite3.IntegrityError:
        return False, "❌ Username or email already exists."
    finally:
        conn.close()

# ✅ Login function (Only authenticate if credentials match)
def login(username, password):
    conn = get_db_connection()
    if not conn:
        return False

    cursor = conn.cursor()
    cursor.execute("SELECT password FROM users WHERE username = ?", (username,))
    user = cursor.fetchone()
    conn.close()
    
    if user and verify_password(password, user[0]):
        st.session_state["authenticated"] = True
        st.session_state["username"] = username
        st.success(f"✅ Welcome, {username}!")
        st.rerun()
        return True
    else:
        st.error("❌ Invalid username or password.")
        return False

# ✅ Logout function
def logout():
    st.session_state["authenticated"] = False
    st.session_state["username"] = ""
    st.success("✅ Logged out successfully.")
    st.rerun()

# ✅ Authentication UI (Handles login/logout properly)
def auth_ui():
    st.sidebar.title("🔐 Authentication")

    # ✅ Ensure session state is initialized properly
    if "authenticated" not in st.session_state:
        st.session_state["authenticated"] = False
        st.session_state["username"] = ""

    # ✅ If user is authenticated, show the logout button
    if st.session_state["authenticated"]:
        st.sidebar.write(f"👤 Logged in as: {st.session_state['username']}")
        if st.sidebar.button("Logout"):
            logout()
        return  # ✅ Prevent showing login/signup UI when logged in

    # ✅ Otherwise, show login/signup options
    menu = ["Login", "Sign Up"]
    choice = st.sidebar.selectbox("Navigation", menu)

    if choice == "Login":
        st.subheader("🔑 Login")
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")

        if st.button("Login"):
            if username and password:
                if login(username, password):
                    st.session_state["authenticated"] = True  
                    st.session_state["username"] = username
                    st.success(f"✅ Welcome, {username}!")
                    st.rerun()
                else:
                    st.error("❌ Invalid username or password.")
            else:
                st.warning("⚠️ Please enter both username and password.")

    elif choice == "Sign Up":
        st.subheader("📝 Create an Account")
        username = st.text_input("Username")
        email = st.text_input("Email")
        password = st.text_input("Password", type="password")

        if st.button("Sign Up"):
            if username and email and password:
                success, message = signup(username, email, password)
                if success:
                    st.success(message)
                else:
                    st.error(message)
            else:
                st.warning("⚠️ All fields are required.")
