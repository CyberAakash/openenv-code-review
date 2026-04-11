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
            _issue(1, "style", "low", "Unused import: 'os' is imported but never used"),
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
            _issue(
                13,
                "style",
                "low",
                "Unused variable: 'x' is assigned but never used",
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
    {
        "task_id": "easy_4",
        "difficulty": "easy",
        "language": "python",
        "code_snippet": """\
import os
import collections
import csv
import io
import time

class dataProcessor:
    '''process data from csv files'''

    def __init__(self, FilePath):
        self.FilePath = FilePath
        self._data = None

    def LoadData(self):
        f = open(self.FilePath, 'r')
        reader = csv.reader(f)
        self._data = list(reader)
        # f is never closed

    def getData(self):
        return self._data

    def processRows(self):
        result = []
        for i in range(0, len(self._data)):
            Row = self._data[i]
            processedRow = [x.strip() for x in Row]
            result.append(processedRow)
        return result

    def SummaryStats(self):
        n = len(self._data)
        print("Total rows: " + str(n))
        AVG_COLS = sum(len(row) for row in self._data) / n
        print("Avg cols: " + str(AVG_COLS))

def Main():
    dp = dataProcessor("data.csv")
    dp.LoadData()
    dp.processRows()
    dp.SummaryStats()

Main()
""",
        "ground_truth": [
            _issue(
                2,
                "style",
                "low",
                "Unused import: 'collections' is imported but never used",
            ),
            _issue(4, "style", "low", "Unused import: 'io' is imported but never used"),
            _issue(
                5, "style", "low", "Unused import: 'time' is imported but never used"
            ),
            _issue(
                7,
                "style",
                "medium",
                "Class name 'dataProcessor' should use PascalCase (PEP 8)",
            ),
            _issue(
                14,
                "style",
                "low",
                "Method name 'LoadData' should use snake_case (PEP 8)",
            ),
            _issue(
                16,
                "bug",
                "medium",
                "Resource leak: file handle 'f' is opened but never closed; use 'with' statement",
            ),
            _issue(
                20,
                "style",
                "low",
                "Method name 'getData' should use snake_case: 'get_data' (PEP 8)",
            ),
            _issue(
                23,
                "style",
                "low",
                "Method name 'processRows' should use snake_case: 'process_rows' (PEP 8)",
            ),
            _issue(
                31,
                "style",
                "low",
                "Method name 'SummaryStats' should use snake_case: 'summary_stats' (PEP 8)",
            ),
        ],
    },
    {
        "task_id": "easy_5",
        "difficulty": "easy",
        "language": "python",
        "code_snippet": """\
import os
import sys
import json
import logging
import re
from typing import Dict, List, Tuple, Optional

logger = logging.getLogger(__name__)

def ReadConfig(filepath: str) -> Dict:
    '''reads config from json file'''
    with open(filepath) as F:
        Data = json.load(F)
    return Data

def parse_Values(raw: str) -> List[int]:
    items = raw.split(",")
    Res = []
    for Item in items:
        val = int(Item.strip())
        Res.append(val)
    return Res

class configManager:
    MAX = 100
    min = 0

    def __init__(self, configPath):
        self.configPath = configPath
        self.Config = ReadConfig(configPath)

    def GetValue(self, key: str):
        return self.Config.get(key, None)

    def set_value(self, key: str, val):
        self.Config[key] = val

    def SaveConfig(self):
        with open(self.configPath, 'w') as f:
            json.dump(self.Config, f, indent=2)

cm = configManager("config.json")
val = cm.GetValue("threshold")
""",
        "ground_truth": [
            _issue(1, "style", "low", "Unused import: 'os' is imported but never used"),
            _issue(
                2, "style", "low", "Unused import: 'sys' is imported but never used"
            ),
            _issue(5, "style", "low", "Unused import: 're' is imported but never used"),
            _issue(
                6,
                "style",
                "low",
                "Unused imports: 'Tuple' and 'Optional' from typing are imported but never used",
            ),
            _issue(
                10,
                "style",
                "low",
                "Function name 'ReadConfig' should use snake_case (PEP 8)",
            ),
            _issue(
                16,
                "style",
                "low",
                "Function name 'parse_Values' has inconsistent casing; should be 'parse_values' (PEP 8)",
            ),
            _issue(
                24,
                "style",
                "medium",
                "Class name 'configManager' should use PascalCase (PEP 8)",
            ),
            _issue(
                26,
                "style",
                "medium",
                "Inconsistent constant naming: 'min' should be uppercase MIN for a class constant",
            ),
            _issue(
                32,
                "style",
                "low",
                "Method name 'GetValue' should use snake_case: 'get_value' (PEP 8)",
            ),
            _issue(
                38,
                "style",
                "low",
                "Method name 'SaveConfig' should use snake_case: 'save_config' (PEP 8)",
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

def safe_get(lst, index, default=None):
    try:
        return lst[index]
    except (IndexError, TypeError):
        return default

def clamp_value(val, lo=0, hi=100):
    return max(lo, min(hi, val))
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
    {
        "task_id": "medium_4",
        "difficulty": "medium",
        "language": "python",
        "code_snippet": """\
import threading

class BankAccount:
    def __init__(self, balance=0):
        self.balance = balance

    def deposit(self, amount):
        self.balance += amount
        return self.balance

    def withdraw(self, amount):
        self.balance -= amount
        return self.balance

    def transfer(self, other, amount):
        self.withdraw(amount)
        other.deposit(amount)

def calculate_discount(price, discount_percent):
    if discount_percent > 1.0:
        discount_percent = discount_percent / 100
    final = price - price * discount_percent
    return round(final)

def find_element(matrix, target):
    for i in range(len(matrix)):
        for j in range(len(matrix[0])):
            if matrix[i][j] == target:
                return (i, j)

def parse_csv_line(line):
    return line.split(",")

def cumulative_sum(numbers):
    result = []
    running = 0
    for n in numbers:
        running += n
    result.append(running)
    return result

def intersect_lists(a, b):
    result = []
    for item in a:
        if item in b:
            result.append(item)
    return result
""",
        "ground_truth": [
            _issue(
                8,
                "bug",
                "high",
                "Race condition: balance += amount is not thread-safe; needs a lock for concurrent access",
            ),
            _issue(
                12,
                "bug",
                "high",
                "No balance check: withdraw allows negative balance; should verify sufficient funds",
            ),
            _issue(
                16,
                "bug",
                "high",
                "Non-atomic transfer: if deposit fails after withdraw, money is lost; needs transaction/lock",
            ),
            _issue(
                24,
                "bug",
                "medium",
                "Rounding error: round(final) rounds to integer losing cents; use round(final, 2) for currency",
            ),
            _issue(
                27,
                "bug",
                "medium",
                "Assumes uniform row length: len(matrix[0]) will fail on empty matrix or jagged arrays",
            ),
            _issue(
                32,
                "bug",
                "medium",
                "Naive CSV parsing: split(',') fails on quoted fields containing commas; use csv module",
            ),
            _issue(
                38,
                "bug",
                "high",
                "Wrong indentation: result.append is outside the loop, only appends final sum instead of cumulative sums",
            ),
            _issue(
                42,
                "bug",
                "medium",
                "Duplicates not handled: if item appears multiple times in both lists, duplicates are included in result",
            ),
        ],
    },
    {
        "task_id": "medium_5",
        "difficulty": "medium",
        "language": "python",
        "code_snippet": """\
from datetime import datetime, timedelta

class Cache:
    def __init__(self, ttl_seconds=300):
        self._store = {}
        self._ttl = ttl_seconds

    def get(self, key):
        if key in self._store:
            return self._store[key]["value"]
        return None

    def set(self, key, value):
        self._store[key] = {
            "value": value,
            "expires": datetime.now() + timedelta(seconds=self._ttl)
        }

    def clear_expired(self):
        for key in self._store:
            if self._store[key]["expires"] < datetime.now():
                del self._store[key]

def levenshtein(s1, s2):
    if len(s1) == 0:
        return len(s2)
    if len(s2) == 0:
        return len(s1)

    matrix = [[0] * (len(s2) + 1)] * (len(s1) + 1)

    for i in range(len(s1) + 1):
        matrix[i][0] = i
    for j in range(len(s2) + 1):
        matrix[0][j] = j

    for i in range(1, len(s1) + 1):
        for j in range(1, len(s2) + 1):
            cost = 0 if s1[i-1] == s2[j-1] else 1
            matrix[i][j] = min(
                matrix[i-1][j] + 1,
                matrix[i][j-1] + 1,
                matrix[i-1][j-1] + cost
            )
    return matrix[len(s1)][len(s2)]

def moving_average(data, window):
    result = []
    for i in range(len(data)):
        chunk = data[i:i+window]
        result.append(sum(chunk) / window)
    return result

def group_by(items, key_func):
    groups = {}
    for item in items:
        key = key_func(item)
        if key not in groups:
            groups[key] = []
        groups[key] = groups[key].append(item)
    return groups
""",
        "ground_truth": [
            _issue(
                10,
                "bug",
                "high",
                "TTL not enforced on read: get() returns expired entries without checking expiration time",
            ),
            _issue(
                20,
                "bug",
                "high",
                "Dict mutation during iteration: deleting keys from self._store while iterating causes RuntimeError",
            ),
            _issue(
                31,
                "bug",
                "critical",
                "Shallow copy trap: [[0] * cols] * rows creates rows that share the same list object; use list comprehension",
            ),
            _issue(
                50,
                "bug",
                "medium",
                "Incomplete window: near the end of data, chunk is smaller than window causing incorrect average divisor",
            ),
            _issue(
                57,
                "bug",
                "high",
                "list.append returns None: 'groups[key] = groups[key].append(item)' sets the value to None; use just groups[key].append(item)",
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

def list_data_directory():
    # Safe: subprocess with list args, no shell injection possible
    result = subprocess.run(["ls", "-la", "/var/data"], capture_output=True, text=True, check=True)
    return result.stdout.splitlines()

def verify_token(provided_token, expected_token):
    return provided_token == expected_token

def get_user_safe(conn, username):
    # Uses parameterized query — safe
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM users WHERE username = ?", (username,))
    return cursor.fetchone()
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
            _issue(
                35,
                "security",
                "medium",
                "Timing side-channel: string comparison with '==' leaks token length via timing; use hmac.compare_digest()",
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

def validate_email(email_str):
    pattern = r"^([a-zA-Z0-9_.+-]+)+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$"
    return bool(re.match(pattern, email_str))

def hash_password_safe(password, salt):
    # Intentionally uses SHA-256 (better than MD5 but still not bcrypt/argon2)
    return hashlib.sha256((salt + password).encode()).hexdigest()
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
            _issue(
                34,
                "security",
                "medium",
                "ReDoS vulnerability: regex pattern with nested quantifiers '([a-zA-Z0-9_.+-]+)+' causes catastrophic backtracking on malicious input",
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
import hashlib
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

def compute_checksum(data):
    # SHA-256 is appropriate for data integrity verification
    return hashlib.sha256(data.encode()).hexdigest()

def generate_password(length=8):
    chars = string.ascii_lowercase
    return "".join(random.choice(chars) for _ in range(length))

def process_user_input(data_str):
    result = eval(data_str)
    return result

def safe_read_file(filepath):
    if os.path.exists(filepath):
        # Time gap between check and use: another process could swap the file
        with open(filepath, "r") as f:
            return f.read()
    return None
""",
        "ground_truth": [
            _issue(
                12,
                "security",
                "critical",
                "Hardcoded secret: API secret key embedded in source code",
            ),
            _issue(
                15,
                "security",
                "critical",
                "Plaintext password comparison: comparing raw password to hash is broken authentication; use proper hash verification",
            ),
            _issue(
                21,
                "security",
                "high",
                "XML External Entity (XXE): 'ET.fromstring()' on untrusted XML can lead to XXE attacks; use defusedxml",
            ),
            _issue(
                25,
                "security",
                "critical",
                "Sensitive data logging: credit card number logged in plaintext; PCI-DSS violation",
            ),
            _issue(
                29,
                "security",
                "high",
                "SSRF vulnerability: 'urllib.request.urlopen(url)' with user-controlled URL allows Server-Side Request Forgery",
            ),
            _issue(
                36,
                "security",
                "medium",
                "Weak password generation: only lowercase ASCII letters, no digits/symbols/uppercase; insufficient entropy",
            ),
            _issue(
                40,
                "security",
                "critical",
                "Code injection: 'eval(data_str)' on untrusted input allows arbitrary code execution",
            ),
            _issue(
                43,
                "security",
                "medium",
                "TOCTOU race condition: os.path.exists() check followed by open() allows symlink attacks between check and use",
            ),
        ],
    },
    {
        "task_id": "hard_4",
        "difficulty": "hard",
        "language": "python",
        "code_snippet": """\
import os
import jwt
import sqlite3
import hashlib
import smtplib
from email.mime.text import MIMEText
import json
import base64
import logging

logger = logging.getLogger(__name__)

JWT_SECRET = "change-me-in-production-12345"
ADMIN_EMAIL = "admin@company.com"
SMTP_PASSWORD = "email-password-not-for-production"

def create_jwt(user_id, role):
    payload = {
        "user_id": user_id,
        "role": role,
    }
    return jwt.encode(payload, JWT_SECRET, algorithm="none")

def verify_jwt(token):
    try:
        return jwt.decode(token, JWT_SECRET, algorithms=["none", "HS256"])
    except:
        return None

def get_user_orders(user_id):
    conn = sqlite3.connect("shop.db")
    cursor = conn.cursor()
    cursor.execute(f"SELECT * FROM orders WHERE user_id = {user_id}")
    return cursor.fetchall()

def reset_password(email, new_password):
    hashed = hashlib.sha1(new_password.encode()).hexdigest()
    conn = sqlite3.connect("shop.db")
    conn.execute(
        f"UPDATE users SET password = '{hashed}' WHERE email = '{email}'"
    )
    conn.commit()

def send_notification(to_email, message):
    msg = MIMEText(message)
    msg["Subject"] = "Notification"
    msg["From"] = ADMIN_EMAIL
    msg["To"] = to_email
    server = smtplib.SMTP("smtp.company.com", 587)
    server.login(ADMIN_EMAIL, SMTP_PASSWORD)
    server.sendmail(ADMIN_EMAIL, to_email, msg.as_string())

def decode_user_data(encoded_str):
    decoded = base64.b64decode(encoded_str)
    return eval(decoded.decode("utf-8"))

def check_permission(user, resource):
    if user.get("role") == "admin":
        return True
    if user.get("user_id") == resource.get("owner_id"):
        return True
    logger.debug(f"Access denied for user {user} on resource {resource}")
    return False
""",
        "ground_truth": [
            _issue(
                13,
                "security",
                "critical",
                "Hardcoded JWT secret: secret key in source code should be loaded from environment variables",
            ),
            _issue(
                15,
                "security",
                "critical",
                "Hardcoded SMTP password: email credentials in source code",
            ),
            _issue(
                22,
                "security",
                "critical",
                "JWT algorithm 'none': signing with algorithm='none' produces unsigned tokens anyone can forge",
            ),
            _issue(
                18,
                "security",
                "high",
                "JWT missing expiration: payload has no 'exp' claim so tokens never expire; compromised tokens are valid forever",
            ),
            _issue(
                26,
                "security",
                "critical",
                "JWT algorithms list includes 'none': allows attackers to submit unsigned tokens that will be accepted",
            ),
            _issue(
                27,
                "bug",
                "medium",
                "Bare except clause: catches all exceptions including KeyboardInterrupt and SystemExit; use specific exception types",
            ),
            _issue(
                33,
                "security",
                "critical",
                "SQL injection: f-string with user_id in query allows SQL injection; use parameterised queries",
            ),
            _issue(
                37,
                "security",
                "high",
                "Weak hash: SHA-1 is cryptographically broken for password hashing; use bcrypt or argon2",
            ),
            _issue(
                40,
                "security",
                "critical",
                "SQL injection: f-string in UPDATE query with email and hashed password; use parameterised queries",
            ),
            _issue(
                49,
                "security",
                "high",
                "Unencrypted SMTP: connecting without STARTTLS sends credentials in plaintext; call server.starttls() first",
            ),
            _issue(
                53,
                "security",
                "critical",
                "Code injection: eval() on decoded user data allows arbitrary code execution",
            ),
            _issue(
                58,
                "security",
                "medium",
                "Sensitive data in logs: logging full user dict may expose passwords/tokens in debug logs",
            ),
        ],
    },
    {
        "task_id": "hard_5",
        "difficulty": "hard",
        "language": "python",
        "code_snippet": """\
import os
import re
import json
import sqlite3
import hashlib
import hmac
import secrets
import subprocess
from http.server import BaseHTTPRequestHandler
from urllib.parse import parse_qs, urlparse
import xml.etree.ElementTree as ET

WEBHOOK_SECRET = "webhook-shared-secret-do-not-leak"
DATABASE_URL = "postgresql://admin:db-pass-12345@prod-db.internal:5432/app"

class RequestHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        parsed = urlparse(self.path)
        params = parse_qs(parsed.query)

        if parsed.path == "/user":
            user_id = params.get("id", [""])[0]
            conn = sqlite3.connect("app.db")
            result = conn.execute(
                "SELECT * FROM users WHERE id = " + user_id
            ).fetchone()
            self.send_response(200)
            self.end_headers()
            self.wfile.write(json.dumps(result).encode())

        elif parsed.path == "/search":
            query = params.get("q", [""])[0]
            self.send_response(200)
            self.end_headers()
            self.wfile.write(f"<html><body>Results for: {query}</body></html>".encode())

        elif parsed.path == "/file":
            filename = params.get("name", [""])[0]
            filepath = os.path.join("/uploads", filename)
            with open(filepath, "rb") as f:
                data = f.read()
            self.send_response(200)
            self.end_headers()
            self.wfile.write(data)

        elif parsed.path == "/export":
            fmt = params.get("format", ["json"])[0]
            cmd = f"export_tool --format {fmt} --output /tmp/export"
            subprocess.call(cmd, shell=True)
            self.send_response(200)
            self.end_headers()

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length).decode()

        if self.path == "/webhook":
            data = json.loads(body)
            self._process_webhook(data)

        elif self.path == "/upload-xml":
            root = ET.fromstring(body)
            self._process_xml(root)

        elif self.path == "/login":
            creds = json.loads(body)
            password_hash = hashlib.md5(creds["password"].encode()).hexdigest()
            conn = sqlite3.connect("app.db")
            user = conn.execute(
                f"SELECT * FROM users WHERE username = '{creds['username']}' "
                f"AND password = '{password_hash}'"
            ).fetchone()
            if user:
                token = secrets.token_hex(16)
                self.send_response(200)
                self.end_headers()
                self.wfile.write(json.dumps({"token": token}).encode())
            else:
                self.send_response(401)
                self.end_headers()

    def _process_webhook(self, data):
        # No signature verification
        action = data.get("action")
        if action:
            os.system(f"process_webhook --action {action}")

    def _process_xml(self, root):
        for elem in root.iter():
            print(elem.tag, elem.text)
""",
        "ground_truth": [
            _issue(
                13,
                "security",
                "critical",
                "Hardcoded webhook secret: shared secret in source code should use environment variables",
            ),
            _issue(
                14,
                "security",
                "critical",
                "Hardcoded database URL with credentials: database password exposed in source code",
            ),
            _issue(
                25,
                "security",
                "critical",
                "SQL injection: string concatenation with user_id in SQL query; use parameterised queries",
            ),
            _issue(
                34,
                "security",
                "high",
                "Reflected XSS: user query parameter rendered directly into HTML response without escaping",
            ),
            _issue(
                39,
                "security",
                "high",
                "Path traversal: user-controlled filename in os.path.join allows reading arbitrary files with '../'",
            ),
            _issue(
                46,
                "security",
                "critical",
                "Command injection: user-controlled format parameter in shell command; use subprocess with list args",
            ),
            _issue(
                60,
                "security",
                "high",
                "XXE vulnerability: ET.fromstring on untrusted XML without disabling external entities; use defusedxml",
            ),
            _issue(
                64,
                "security",
                "high",
                "Weak password hashing: MD5 is cryptographically broken for passwords; use bcrypt or argon2",
            ),
            _issue(
                66,
                "security",
                "critical",
                "SQL injection: f-string with username and password in login query; use parameterised queries",
            ),
            _issue(
                78,
                "security",
                "critical",
                "Missing webhook signature verification: webhook data is processed without HMAC verification",
            ),
            _issue(
                80,
                "security",
                "critical",
                "Command injection: os.system with user-controlled action from webhook data; use subprocess with list args",
            ),
            _issue(
                2,
                "style",
                "low",
                "Unused import: 're' is imported but never used in this module",
            ),
            _issue(
                6,
                "style",
                "low",
                "Unused import: 'hmac' is imported but never used; WEBHOOK_SECRET exists but no HMAC verification is implemented",
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
