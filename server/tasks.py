"""Task definitions for the Code Review environment.

Each task contains:
- task_id: unique identifier
- difficulty: easy | medium | hard
- code_snippet: Python source code to review
- ground_truth: list of annotated issues the agent should find
"""

from typing import Any, Dict, List

# ---------------------------------------------------------------------------
# Ground-truth issue helper
# ---------------------------------------------------------------------------


def _issue(
    line: int, issue_type: str, severity: str, description: str
) -> Dict[str, Any]:
    return {
        "line_number": line,
        "issue_type": issue_type,
        "severity": severity,
        "description": description,
    }


# ===================================================================
# EASY TASKS  — Style & syntax issues (3-4 issues per snippet)
# ===================================================================

EASY_TASKS: List[Dict[str, Any]] = [
    {
        "task_id": "easy_1",
        "difficulty": "easy",
        "language": "python",
        "code_snippet": """\
import os
import sys
import json
import math

def calcArea(r):
    area = 3.14159 * r * r
    return area

def process_data(data):
    result = []
    for i in range(len(data)):
        result.append(data[i] * 2)
    return result

x = 100
print(calcArea(5))
""",
        "ground_truth": [
            _issue(
                2, "style", "low", "Unused import: 'sys' is imported but never used"
            ),
            _issue(
                3, "style", "low", "Unused import: 'json' is imported but never used"
            ),
            _issue(
                4, "style", "low", "Unused import: 'math' is imported but never used"
            ),
            _issue(
                7,
                "style",
                "medium",
                "Magic number: 3.14159 should be replaced with math.pi",
            ),
            _issue(
                6,
                "style",
                "low",
                "Function name 'calcArea' should use snake_case (PEP 8)",
            ),
        ],
    },
    {
        "task_id": "easy_2",
        "difficulty": "easy",
        "language": "python",
        "code_snippet": """\
import re
import datetime

class userProfile:
    def __init__(self, Name, age, email):
        self.Name = Name
        self.age = age
        self.email = email

    def GetAge(self):
        return self.age

    def display(self):
        print("Name: " + self.Name + " Age: " + str(self.age))

def validate_email(email):
    # TODO: implement validation
    pass

u = userProfile("Alice", 30, "alice@example.com")
u.display()
""",
        "ground_truth": [
            _issue(1, "style", "low", "Unused import: 're' is imported but never used"),
            _issue(
                2,
                "style",
                "low",
                "Unused import: 'datetime' is imported but never used",
            ),
            _issue(
                4,
                "style",
                "medium",
                "Class name 'userProfile' should use CapWords/PascalCase (PEP 8)",
            ),
            _issue(
                10, "style", "low", "Method name 'GetAge' should use snake_case (PEP 8)"
            ),
        ],
    },
    {
        "task_id": "easy_3",
        "difficulty": "easy",
        "language": "python",
        "code_snippet": """\
from typing import List
import os, sys, json

def multiply_list(lst: List[int]) -> List[int]:
    RES = []
    for i in range(0, len(lst)):
        RES.append(lst[i] * 2)
    return RES

def connect_to_db():
    host = "localhost"
    port = 5432
    timeout = 30
    retries = 3
    # actual connection logic omitted
    pass

print(multiply_list([1, 2, 3]))
""",
        "ground_truth": [
            _issue(
                2,
                "style",
                "low",
                "Multiple imports on one line: 'os, sys, json' should be separate import statements",
            ),
            _issue(
                2,
                "style",
                "low",
                "Unused imports: 'os', 'sys', 'json' are imported but never used",
            ),
            _issue(
                5,
                "style",
                "low",
                "Variable name 'RES' uses ALL_CAPS; should be lowercase (PEP 8 — ALL_CAPS is for constants)",
            ),
            _issue(
                12,
                "style",
                "medium",
                "Magic number: 5432 should be a named constant (e.g. DEFAULT_PORT)",
            ),
        ],
    },
]


# ===================================================================
# MEDIUM TASKS  — Logic bugs (4-5 issues per snippet)
# ===================================================================

MEDIUM_TASKS: List[Dict[str, Any]] = [
    {
        "task_id": "medium_1",
        "difficulty": "medium",
        "language": "python",
        "code_snippet": """\
def binary_search(arr, target):
    low = 0
    high = len(arr)
    while low < high:
        mid = (low + high) / 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            low = mid
        else:
            high = mid - 1
    return -1

def average(numbers):
    total = 0
    for n in numbers:
        total += n
    return total / len(numbers)

def find_max(lst):
    max_val = 0
    for item in lst:
        if item > max_val:
            max_val = item
    return max_val

def is_palindrome(s):
    s = s.lower()
    return s == s[::-1]
""",
        "ground_truth": [
            _issue(
                3,
                "bug",
                "high",
                "Off-by-one: 'high = len(arr)' should be 'len(arr) - 1' for inclusive binary search, or adjust the while condition",
            ),
            _issue(
                5,
                "bug",
                "high",
                "Integer division needed: '(low + high) / 2' produces float; use '//' for integer index",
            ),
            _issue(
                9,
                "bug",
                "high",
                "Infinite loop: 'low = mid' should be 'low = mid + 1' to make progress",
            ),
            _issue(
                18,
                "bug",
                "high",
                "ZeroDivisionError: 'len(numbers)' will crash when the list is empty — missing empty-list check",
            ),
            _issue(
                21,
                "bug",
                "medium",
                "Incorrect initialisation: 'max_val = 0' fails for lists with all negative numbers — use float('-inf') or lst[0]",
            ),
        ],
    },
    {
        "task_id": "medium_2",
        "difficulty": "medium",
        "language": "python",
        "code_snippet": """\
def flatten(nested_list):
    result = []
    for item in nested_list:
        if type(item) == list:
            flatten(item)
        else:
            result.append(item)
    return result

def count_words(text):
    words = text.split(" ")
    counts = {}
    for word in words:
        if word in counts:
            counts[word] = +1
        else:
            counts[word] = 1
    return counts

def safe_divide(a, b):
    if b == 0:
        return 0
    return a / b

def remove_duplicates(lst):
    seen = set()
    result = []
    for item in lst:
        if item in seen:
            result.append(item)
        seen.add(item)
    return result
""",
        "ground_truth": [
            _issue(
                5,
                "bug",
                "high",
                "Return value discarded: recursive call 'flatten(item)' result is not added to 'result'",
            ),
            _issue(
                4,
                "bug",
                "medium",
                "Use isinstance() instead of 'type(item) == list' for proper subclass handling",
            ),
            _issue(
                15,
                "bug",
                "high",
                "Wrong operator: 'counts[word] = +1' sets value to 1 (unary +), should be 'counts[word] += 1'",
            ),
            _issue(
                22,
                "bug",
                "medium",
                "Silent error: returning 0 on division-by-zero hides the error; consider returning None or raising ValueError",
            ),
            _issue(
                30,
                "bug",
                "high",
                "Inverted logic: appends item when it IS in 'seen' (duplicates only); should append when NOT in seen",
            ),
        ],
    },
    {
        "task_id": "medium_3",
        "difficulty": "medium",
        "language": "python",
        "code_snippet": """\
def merge_sorted(a, b):
    result = []
    i, j = 0, 0
    while i < len(a) and j < len(b):
        if a[i] <= b[j]:
            result.append(a[i])
            i += 1
        else:
            result.append(b[j])
            j += 1
    return result

def parse_int(s):
    try:
        return int(s)
    except:
        return s

def fibonacci(n):
    if n <= 0:
        return 0
    if n == 1:
        return 1
    return fibonacci(n - 1) + fibonacci(n - 2)

def clamp(value, min, max):
    if value < min:
        return min
    if value > max:
        return max
    return value
""",
        "ground_truth": [
            _issue(
                11,
                "bug",
                "high",
                "Missing remaining elements: after the while loop, remaining elements of 'a' or 'b' are not appended to result",
            ),
            _issue(
                16,
                "bug",
                "medium",
                "Bare except: catches all exceptions including KeyboardInterrupt and SystemExit; use 'except ValueError'",
            ),
            _issue(
                17,
                "bug",
                "medium",
                "Returning original string on parse failure silently changes return type; caller expects int",
            ),
            _issue(
                26,
                "bug",
                "low",
                "Shadowing built-ins: parameters 'min' and 'max' shadow Python built-in functions",
            ),
        ],
    },
]


# ===================================================================
# HARD TASKS  — Security vulnerabilities (5-6 issues per snippet)
# ===================================================================

HARD_TASKS: List[Dict[str, Any]] = [
    {
        "task_id": "hard_1",
        "difficulty": "hard",
        "language": "python",
        "code_snippet": """\
import os
import sqlite3
import subprocess
import pickle
import random

SECRET_KEY = "hardcoded-api-secret-key-12345-do-not-commit"

def get_user(username):
    conn = sqlite3.connect("app.db")
    cursor = conn.cursor()
    query = f"SELECT * FROM users WHERE username = '{username}'"
    cursor.execute(query)
    return cursor.fetchone()

def run_command(user_input):
    result = os.system("echo " + user_input)
    return result

def load_user_data(data_bytes):
    return pickle.loads(data_bytes)

def generate_token():
    return str(random.randint(100000, 999999))

def read_file(filename):
    path = "/var/data/" + filename
    with open(path, "r") as f:
        return f.read()
""",
        "ground_truth": [
            _issue(
                7,
                "security",
                "critical",
                "Hardcoded secret: API key is embedded in source code; use environment variables",
            ),
            _issue(
                12,
                "security",
                "critical",
                "SQL injection: f-string query with unsanitised 'username'; use parameterised queries",
            ),
            _issue(
                17,
                "security",
                "critical",
                "Command injection: 'os.system(\"echo \" + user_input)' allows arbitrary command execution; use subprocess with list args",
            ),
            _issue(
                21,
                "security",
                "high",
                "Insecure deserialisation: 'pickle.loads(data_bytes)' on untrusted input allows arbitrary code execution",
            ),
            _issue(
                24,
                "security",
                "high",
                "Insecure random: 'random.randint' is not cryptographically secure; use 'secrets' module for tokens",
            ),
            _issue(
                27,
                "security",
                "high",
                "Path traversal: user-controlled 'filename' concatenated to path; attacker can use '../' to read arbitrary files",
            ),
        ],
    },
    {
        "task_id": "hard_2",
        "difficulty": "hard",
        "language": "python",
        "code_snippet": """\
import hashlib
import yaml
import tempfile
import re
import subprocess

DB_PASSWORD = "admin123"

def hash_password(password):
    return hashlib.md5(password.encode()).hexdigest()

def load_config(user_yaml):
    return yaml.load(user_yaml)

def run_lint(filename):
    cmd = "pylint " + filename
    return subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE).communicate()

def check_admin(request):
    token = request.cookies.get("admin_token")
    if token == "super_secret_admin":
        return True
    return False

def create_temp_file(content):
    path = tempfile.mktemp()
    with open(path, "w") as f:
        f.write(content)
    return path

def sanitize_html(html_input):
    clean = re.sub(r"<script>", "", html_input)
    return clean
""",
        "ground_truth": [
            _issue(
                7,
                "security",
                "critical",
                "Hardcoded credential: 'DB_PASSWORD' with plaintext password in source code",
            ),
            _issue(
                10,
                "security",
                "high",
                "Weak hash algorithm: MD5 is cryptographically broken for password hashing; use bcrypt or argon2",
            ),
            _issue(
                13,
                "security",
                "critical",
                "Unsafe YAML loading: 'yaml.load()' without Loader allows arbitrary code execution; use 'yaml.safe_load()'",
            ),
            _issue(
                17,
                "security",
                "critical",
                "Command injection: 'shell=True' with string concatenation allows arbitrary command execution",
            ),
            _issue(
                21,
                "security",
                "high",
                "Hardcoded token comparison: admin check uses a magic string; use proper authentication",
            ),
            _issue(
                26,
                "security",
                "medium",
                "Insecure temp file: 'tempfile.mktemp()' is vulnerable to race conditions; use 'tempfile.mkstemp()'",
            ),
            _issue(
                31,
                "security",
                "high",
                "Incomplete HTML sanitisation: only removes '<script>' tag; does not handle variations like '<SCRIPT>', '<script >', event handlers, etc.",
            ),
        ],
    },
    {
        "task_id": "hard_3",
        "difficulty": "hard",
        "language": "python",
        "code_snippet": """\
import os
import json
import logging
import xml.etree.ElementTree as ET
import random
import string

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

API_SECRET = "my-super-secret-api-key-do-not-share-12345"

def authenticate(user_password, stored_hash):
    if user_password == stored_hash:
        return True
    return False

def parse_xml(xml_string):
    root = ET.fromstring(xml_string)
    return root

def log_user_action(username, action, credit_card):
    logger.info(f"User {username} performed {action} with card {credit_card}")

def make_request(url):
    import urllib.request
    response = urllib.request.urlopen(url)
    return response.read()

def generate_password(length=8):
    chars = string.ascii_lowercase
    return "".join(random.choice(chars) for _ in range(length))

def process_user_input(data_str):
    result = eval(data_str)
    return result
""",
        "ground_truth": [
            _issue(
                11,
                "security",
                "critical",
                "Hardcoded secret: GitHub personal access token in source code",
            ),
            _issue(
                14,
                "security",
                "critical",
                "Plaintext password comparison: comparing raw password to hash is broken authentication; use proper hash verification",
            ),
            _issue(
                20,
                "security",
                "high",
                "XML External Entity (XXE): 'ET.fromstring()' on untrusted XML can lead to XXE attacks; use defusedxml",
            ),
            _issue(
                24,
                "security",
                "critical",
                "Sensitive data logging: credit card number logged in plaintext; PCI-DSS violation",
            ),
            _issue(
                28,
                "security",
                "high",
                "SSRF vulnerability: 'urllib.request.urlopen(url)' with user-controlled URL allows Server-Side Request Forgery",
            ),
            _issue(
                32,
                "security",
                "medium",
                "Weak password generation: only lowercase ASCII letters, no digits/symbols/uppercase; insufficient entropy",
            ),
            _issue(
                35,
                "security",
                "critical",
                "Code injection: 'eval(data_str)' on untrusted input allows arbitrary code execution",
            ),
        ],
    },
]


# ===================================================================
# Task registry
# ===================================================================

ALL_TASKS: Dict[str, Dict[str, Any]] = {}
for _task in EASY_TASKS + MEDIUM_TASKS + HARD_TASKS:
    ALL_TASKS[_task["task_id"]] = _task

# Quick-access lists by difficulty
TASKS_BY_DIFFICULTY: Dict[str, List[str]] = {
    "easy": [t["task_id"] for t in EASY_TASKS],
    "medium": [t["task_id"] for t in MEDIUM_TASKS],
    "hard": [t["task_id"] for t in HARD_TASKS],
}


def get_task(task_id: str) -> Dict[str, Any]:
    """Return a task dict by its ID, or raise KeyError."""
    if task_id not in ALL_TASKS:
        available = ", ".join(sorted(ALL_TASKS.keys()))
        raise KeyError(f"Unknown task_id '{task_id}'. Available: {available}")
    return ALL_TASKS[task_id]


def list_task_ids() -> List[str]:
    """Return all available task IDs."""
    return sorted(ALL_TASKS.keys())
