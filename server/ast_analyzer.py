"""Real AST-based code analysis module.

Implements Bandit-inspired security/style/bug checks using Python's ast module.
Each check walks the AST and returns findings with line numbers, types, severity,
descriptions, and confidence scores.

Also provides:
- get_ast_summary(): structural metadata for observations
- verify_fix(): validates agent-submitted code fixes

Check mapping (modeled after Bandit):
  B102  -> eval()/exec() calls          (code injection)
  B602  -> subprocess(shell=True)        (command injection)
  B605  -> os.system() calls             (command injection)
  B301  -> pickle.loads()                (insecure deserialization)
  B608  -> f-string/format in SQL        (SQL injection)
  B110  -> bare except:                  (swallowed exceptions)
  B105  -> hardcoded passwords/secrets   (credential leaks)
  B506  -> yaml.load() without SafeLoader (unsafe YAML)
  custom -> random for security contexts (weak randomness)
  custom -> file open without 'with'     (resource leak)
  custom -> unused imports               (style)
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple


# ---------------------------------------------------------------------------
# Data classes for findings and fix results
# ---------------------------------------------------------------------------


@dataclass
class ASTFinding:
    """A single finding from AST analysis."""

    line_number: int
    issue_type: str  # "security" | "bug" | "style"
    severity: str  # "low" | "medium" | "high" | "critical"
    description: str
    confidence: float  # 0.0 to 1.0 — how confident the check is
    check_id: str  # e.g. "B102", "B602", etc.

    def to_dict(self) -> Dict[str, Any]:
        return {
            "line_number": self.line_number,
            "issue_type": self.issue_type,
            "severity": self.severity,
            "description": self.description,
            "confidence": self.confidence,
            "check_id": self.check_id,
        }


@dataclass
class FixResult:
    """Result of verifying a submitted code fix."""

    is_valid: bool
    score: float  # 0.0 to 1.0
    feedback: str
    issues_introduced: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Dangerous function/module sets
# ---------------------------------------------------------------------------

DANGEROUS_IMPORTS = {
    "os",
    "subprocess",
    "pickle",
    "marshal",
    "shelve",
    "tempfile",
    "shutil",
    "ctypes",
    "xml.etree.ElementTree",
}

HARDCODED_SECRET_PATTERNS = re.compile(
    r"(password|passwd|secret|api_key|apikey|token|private_key|"
    r"access_key|auth_token|credentials?)\s*=\s*['\"][^'\"]{4,}['\"]",
    re.IGNORECASE,
)

SECRET_VAR_NAMES = re.compile(
    r"(password|passwd|secret|api_key|apikey|token|private_key|"
    r"access_key|auth_token|credentials?|secret_key|db_password)",
    re.IGNORECASE,
)

SQL_KEYWORDS = re.compile(
    r"\b(SELECT|INSERT|UPDATE|DELETE|DROP|CREATE|ALTER|EXEC|EXECUTE|UNION)\b",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# AST Check functions
# ---------------------------------------------------------------------------


def _check_eval_exec(tree: ast.AST) -> List[ASTFinding]:
    """B102: Detect eval() and exec() calls — code injection risk."""
    findings = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func_name = _get_call_name(node)
            if func_name in ("eval", "exec"):
                findings.append(
                    ASTFinding(
                        line_number=node.lineno,
                        issue_type="security",
                        severity="critical",
                        description=f"Use of {func_name}() is a code injection risk. "
                        f"Avoid executing dynamically constructed code.",
                        confidence=0.95,
                        check_id="B102",
                    )
                )
    return findings


def _check_os_system(tree: ast.AST) -> List[ASTFinding]:
    """B605: Detect os.system() calls — command injection risk."""
    findings = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func_name = _get_call_name(node)
            if func_name in ("os.system", "os.popen"):
                findings.append(
                    ASTFinding(
                        line_number=node.lineno,
                        issue_type="security",
                        severity="high",
                        description=f"Use of {func_name}() allows command injection. "
                        f"Use subprocess.run() with a list of arguments instead.",
                        confidence=0.90,
                        check_id="B605",
                    )
                )
    return findings


def _check_subprocess_shell(tree: ast.AST) -> List[ASTFinding]:
    """B602: Detect subprocess calls with shell=True — command injection risk."""
    findings = []
    subprocess_funcs = {
        "subprocess.call",
        "subprocess.run",
        "subprocess.Popen",
        "subprocess.check_output",
        "subprocess.check_call",
    }
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func_name = _get_call_name(node)
            if func_name in subprocess_funcs:
                for kw in node.keywords:
                    if kw.arg == "shell":
                        if (
                            isinstance(kw.value, ast.Constant)
                            and kw.value.value is True
                        ):
                            findings.append(
                                ASTFinding(
                                    line_number=node.lineno,
                                    issue_type="security",
                                    severity="high",
                                    description=f"{func_name}() called with shell=True "
                                    f"allows shell injection. Use shell=False with a list of args.",
                                    confidence=0.95,
                                    check_id="B602",
                                )
                            )
                        elif (
                            isinstance(kw.value, ast.NameConstant)
                            and kw.value.value is True
                        ):
                            # Python 3.7 compat
                            findings.append(
                                ASTFinding(
                                    line_number=node.lineno,
                                    issue_type="security",
                                    severity="high",
                                    description=f"{func_name}() called with shell=True "
                                    f"allows shell injection. Use shell=False with a list of args.",
                                    confidence=0.95,
                                    check_id="B602",
                                )
                            )
    return findings


def _check_pickle(tree: ast.AST) -> List[ASTFinding]:
    """B301: Detect pickle.loads/load — insecure deserialization."""
    findings = []
    dangerous_calls = {
        "pickle.loads",
        "pickle.load",
        "cPickle.loads",
        "cPickle.load",
        "marshal.loads",
        "marshal.load",
        "shelve.open",
    }
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func_name = _get_call_name(node)
            if func_name in dangerous_calls:
                findings.append(
                    ASTFinding(
                        line_number=node.lineno,
                        issue_type="security",
                        severity="high",
                        description=f"Use of {func_name}() allows insecure deserialization. "
                        f"Untrusted data could execute arbitrary code. Use json or safe alternatives.",
                        confidence=0.90,
                        check_id="B301",
                    )
                )
    return findings


def _check_sql_injection(tree: ast.AST, source_lines: List[str]) -> List[ASTFinding]:
    """B608: Detect SQL queries built with string formatting/f-strings."""
    findings = []

    for node in ast.walk(tree):
        # Check f-strings that look like SQL
        if isinstance(node, ast.JoinedStr):
            # Reconstruct the f-string approximately from source
            if node.lineno and node.lineno <= len(source_lines):
                line_text = source_lines[node.lineno - 1]
                if SQL_KEYWORDS.search(line_text):
                    findings.append(
                        ASTFinding(
                            line_number=node.lineno,
                            issue_type="security",
                            severity="critical",
                            description="SQL query built with f-string is vulnerable to SQL injection. "
                            "Use parameterized queries instead.",
                            confidence=0.85,
                            check_id="B608",
                        )
                    )

        # Check string formatting with % or .format() that looks like SQL
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mod):
            if isinstance(node.left, ast.Constant) and isinstance(node.left.value, str):
                if SQL_KEYWORDS.search(node.left.value):
                    findings.append(
                        ASTFinding(
                            line_number=node.lineno,
                            issue_type="security",
                            severity="critical",
                            description="SQL query built with string formatting (%) is vulnerable "
                            "to SQL injection. Use parameterized queries instead.",
                            confidence=0.90,
                            check_id="B608",
                        )
                    )

        # Check .format() on SQL strings
        if isinstance(node, ast.Call):
            if (
                isinstance(node.func, ast.Attribute)
                and node.func.attr == "format"
                and isinstance(node.func.value, ast.Constant)
                and isinstance(node.func.value.value, str)
            ):
                if SQL_KEYWORDS.search(node.func.value.value):
                    findings.append(
                        ASTFinding(
                            line_number=node.lineno,
                            issue_type="security",
                            severity="critical",
                            description="SQL query built with .format() is vulnerable to SQL injection. "
                            "Use parameterized queries instead.",
                            confidence=0.90,
                            check_id="B608",
                        )
                    )

        # Check string concatenation with + in cursor.execute()
        if isinstance(node, ast.Call):
            func_name = _get_call_name(node)
            if func_name and "execute" in func_name:
                for arg in node.args:
                    if isinstance(arg, ast.BinOp) and isinstance(arg.op, ast.Add):
                        if _contains_string_with_sql(arg):
                            findings.append(
                                ASTFinding(
                                    line_number=node.lineno,
                                    issue_type="security",
                                    severity="critical",
                                    description="SQL query built with string concatenation in execute() "
                                    "is vulnerable to SQL injection. Use parameterized queries.",
                                    confidence=0.85,
                                    check_id="B608",
                                )
                            )

    return findings


def _check_bare_except(tree: ast.AST) -> List[ASTFinding]:
    """B110: Detect bare 'except:' clauses that swallow all exceptions."""
    findings = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ExceptHandler):
            if node.type is None:
                findings.append(
                    ASTFinding(
                        line_number=node.lineno,
                        issue_type="bug",
                        severity="medium",
                        description="Bare 'except:' catches all exceptions including "
                        "KeyboardInterrupt and SystemExit. Catch specific exceptions instead.",
                        confidence=0.95,
                        check_id="B110",
                    )
                )
    return findings


def _check_hardcoded_secrets(
    tree: ast.AST, source_lines: List[str]
) -> List[ASTFinding]:
    """B105-B107: Detect hardcoded passwords, secrets, API keys."""
    findings = []
    seen_lines: Set[int] = set()

    for node in ast.walk(tree):
        # Check assignments like password = "secret123"
        if isinstance(node, ast.Assign):
            for target in node.targets:
                var_name = ""
                if isinstance(target, ast.Name):
                    var_name = target.id
                elif isinstance(target, ast.Attribute):
                    var_name = target.attr

                if var_name and SECRET_VAR_NAMES.search(var_name):
                    if isinstance(node.value, ast.Constant) and isinstance(
                        node.value.value, str
                    ):
                        if len(node.value.value) >= 4 and node.lineno not in seen_lines:
                            seen_lines.add(node.lineno)
                            findings.append(
                                ASTFinding(
                                    line_number=node.lineno,
                                    issue_type="security",
                                    severity="high",
                                    description=f"Hardcoded secret in variable '{var_name}'. "
                                    f"Use environment variables or a secrets manager instead.",
                                    confidence=0.85,
                                    check_id="B105",
                                )
                            )

        # Check keyword arguments like connect(password="secret123")
        if isinstance(node, ast.Call):
            for kw in node.keywords:
                if kw.arg and SECRET_VAR_NAMES.search(kw.arg):
                    if isinstance(kw.value, ast.Constant) and isinstance(
                        kw.value.value, str
                    ):
                        if len(kw.value.value) >= 4 and node.lineno not in seen_lines:
                            seen_lines.add(node.lineno)
                            findings.append(
                                ASTFinding(
                                    line_number=node.lineno,
                                    issue_type="security",
                                    severity="high",
                                    description=f"Hardcoded secret in keyword argument '{kw.arg}'. "
                                    f"Use environment variables or a secrets manager.",
                                    confidence=0.80,
                                    check_id="B106",
                                )
                            )

    return findings


def _check_yaml_load(tree: ast.AST) -> List[ASTFinding]:
    """B506: Detect yaml.load() without SafeLoader."""
    findings = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func_name = _get_call_name(node)
            if func_name in ("yaml.load", "yaml.unsafe_load"):
                # Check if Loader=SafeLoader is specified
                has_safe_loader = False
                for kw in node.keywords:
                    if kw.arg == "Loader":
                        if isinstance(kw.value, ast.Attribute):
                            if kw.value.attr in ("SafeLoader", "CSafeLoader"):
                                has_safe_loader = True
                        elif isinstance(kw.value, ast.Name):
                            if kw.value.id in ("SafeLoader", "CSafeLoader"):
                                has_safe_loader = True

                if not has_safe_loader:
                    findings.append(
                        ASTFinding(
                            line_number=node.lineno,
                            issue_type="security",
                            severity="high",
                            description=f"Use of {func_name}() without SafeLoader allows "
                            f"arbitrary code execution. Use yaml.safe_load() or "
                            f"yaml.load(data, Loader=yaml.SafeLoader) instead.",
                            confidence=0.95,
                            check_id="B506",
                        )
                    )
    return findings


def _check_weak_randomness(tree: ast.AST, source_lines: List[str]) -> List[ASTFinding]:
    """Detect use of random module for security-sensitive contexts."""
    findings = []
    random_calls = {
        "random.randint",
        "random.random",
        "random.choice",
        "random.randrange",
        "random.sample",
    }
    # Heuristic: look for random usage near security-related variable names
    security_context_keywords = {
        "token",
        "secret",
        "password",
        "key",
        "salt",
        "nonce",
        "otp",
        "session",
        "csrf",
        "auth",
    }

    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func_name = _get_call_name(node)
            if func_name in random_calls:
                # Check surrounding context for security-related usage
                if node.lineno and node.lineno <= len(source_lines):
                    # Check the current line and a few surrounding lines for context
                    start = max(0, node.lineno - 3)
                    end = min(len(source_lines), node.lineno + 2)
                    context = " ".join(source_lines[start:end]).lower()
                    if any(kw in context for kw in security_context_keywords):
                        findings.append(
                            ASTFinding(
                                line_number=node.lineno,
                                issue_type="security",
                                severity="medium",
                                description=f"Use of {func_name}() in a security-sensitive context. "
                                f"The random module is not cryptographically secure. "
                                f"Use secrets module or os.urandom() instead.",
                                confidence=0.70,
                                check_id="WEAK_RANDOM",
                            )
                        )
    return findings


def _check_file_no_with(tree: ast.AST) -> List[ASTFinding]:
    """Detect file open() calls not used with 'with' statement (resource leak)."""
    findings = []

    # Collect line numbers of open() calls inside 'with' statements
    with_open_lines: Set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.With):
            for item in node.items:
                if isinstance(item.context_expr, ast.Call):
                    call_name = _get_call_name(item.context_expr)
                    if call_name in ("open", "builtins.open", "io.open"):
                        with_open_lines.add(item.context_expr.lineno)

    # Now find open() calls NOT in with statements
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func_name = _get_call_name(node)
            if func_name in ("open", "builtins.open", "io.open"):
                if node.lineno not in with_open_lines:
                    findings.append(
                        ASTFinding(
                            line_number=node.lineno,
                            issue_type="bug",
                            severity="medium",
                            description="File opened without 'with' statement. "
                            "This may lead to resource leaks if the file is not properly closed. "
                            "Use 'with open(...) as f:' instead.",
                            confidence=0.80,
                            check_id="RESOURCE_LEAK",
                        )
                    )
    return findings


def _check_unused_imports(tree: ast.AST) -> List[ASTFinding]:
    """Detect imports that are never referenced in the code body."""
    findings = []

    # Collect all imports
    imports: List[Tuple[str, str, int]] = []  # (imported_name, module, lineno)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                name = alias.asname if alias.asname else alias.name
                imports.append((name, alias.name, node.lineno))
        elif isinstance(node, ast.ImportFrom):
            for alias in node.names:
                name = alias.asname if alias.asname else alias.name
                imports.append((name, f"{node.module}.{alias.name}", node.lineno))

    # Collect all Name references (excluding import lines themselves)
    import_lines = {lineno for _, _, lineno in imports}
    used_names: Set[str] = set()
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Name)
            and getattr(node, "lineno", 0) not in import_lines
        ):
            used_names.add(node.id)
        elif isinstance(node, ast.Attribute):
            # Walk up to get the root name
            root = node
            while isinstance(root, ast.Attribute):
                root = root.value
            if (
                isinstance(root, ast.Name)
                and getattr(root, "lineno", 0) not in import_lines
            ):
                used_names.add(root.id)

    for name, module, lineno in imports:
        # Check if the imported name is used
        # For dotted imports like 'import os', check if 'os' appears as a Name
        base_name = name.split(".")[0]
        if base_name not in used_names:
            findings.append(
                ASTFinding(
                    line_number=lineno,
                    issue_type="style",
                    severity="low",
                    description=f"Unused import: '{name}' is imported but never used in the code.",
                    confidence=0.85,
                    check_id="UNUSED_IMPORT",
                )
            )

    return findings


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------


def _get_call_name(node: ast.Call) -> str:
    """Extract the full dotted name from a Call node (e.g. 'os.system')."""
    if isinstance(node.func, ast.Name):
        return node.func.id
    elif isinstance(node.func, ast.Attribute):
        parts = []
        current = node.func
        while isinstance(current, ast.Attribute):
            parts.append(current.attr)
            current = current.value
        if isinstance(current, ast.Name):
            parts.append(current.id)
        return ".".join(reversed(parts))
    return ""


def _contains_string_with_sql(node: ast.AST) -> bool:
    """Check if an AST node contains a string constant with SQL keywords."""
    for child in ast.walk(node):
        if isinstance(child, ast.Constant) and isinstance(child.value, str):
            if SQL_KEYWORDS.search(child.value):
                return True
    return False


# ---------------------------------------------------------------------------
# Main analysis entry point
# ---------------------------------------------------------------------------


ALL_CHECKS = [
    _check_eval_exec,
    _check_os_system,
    _check_subprocess_shell,
    _check_pickle,
    _check_bare_except,
    _check_yaml_load,
    _check_file_no_with,
    _check_unused_imports,
]

# Checks that need source_lines
CHECKS_WITH_SOURCE = [
    _check_sql_injection,
    _check_hardcoded_secrets,
    _check_weak_randomness,
]


def analyze_code(code_snippet: str) -> List[ASTFinding]:
    """Run all AST-based checks on a code snippet.

    Returns a list of ASTFinding objects. If the code cannot be parsed,
    returns a single finding indicating a syntax error.
    """
    try:
        tree = ast.parse(code_snippet)
    except SyntaxError as e:
        return [
            ASTFinding(
                line_number=e.lineno or 1,
                issue_type="bug",
                severity="high",
                description=f"Syntax error: {e.msg}",
                confidence=1.0,
                check_id="SYNTAX_ERROR",
            )
        ]

    source_lines = code_snippet.splitlines()
    findings: List[ASTFinding] = []

    # Run checks that only need the tree
    for check_fn in ALL_CHECKS:
        try:
            findings.extend(check_fn(tree))
        except Exception:
            pass  # Individual check failures shouldn't break the whole analysis

    # Run checks that need source lines
    for check_fn in CHECKS_WITH_SOURCE:
        try:
            findings.extend(check_fn(tree, source_lines))
        except Exception:
            pass

    # Sort by line number for consistent output
    findings.sort(key=lambda f: f.line_number)
    return findings


# ---------------------------------------------------------------------------
# AST Summary (Feature C)
# ---------------------------------------------------------------------------


def get_ast_summary(code_snippet: str) -> Dict[str, Any]:
    """Extract structural metadata from code for the observation.

    Returns a summary dict with function names, imports, line count,
    call count, class count, and count of dangerous imports.
    """
    try:
        tree = ast.parse(code_snippet)
    except SyntaxError:
        return {
            "functions": [],
            "classes": [],
            "imports": [],
            "total_lines": len(code_snippet.splitlines()),
            "call_count": 0,
            "class_count": 0,
            "dangerous_import_count": 0,
            "parse_error": True,
        }

    functions: List[str] = []
    classes: List[str] = []
    imports: List[str] = []
    call_count = 0

    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) or isinstance(node, ast.AsyncFunctionDef):
            functions.append(node.name)
        elif isinstance(node, ast.ClassDef):
            classes.append(node.name)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                imports.append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                imports.append(node.module)
        elif isinstance(node, ast.Call):
            call_count += 1

    # Count dangerous imports
    dangerous_count = sum(
        1
        for imp in imports
        if imp in DANGEROUS_IMPORTS or imp.split(".")[0] in DANGEROUS_IMPORTS
    )

    return {
        "functions": functions,
        "classes": classes,
        "imports": imports,
        "total_lines": len(code_snippet.splitlines()),
        "call_count": call_count,
        "class_count": len(classes),
        "dangerous_import_count": dangerous_count,
    }


# ---------------------------------------------------------------------------
# Fix Verification (Feature B)
# ---------------------------------------------------------------------------

# Maps check_id to (dangerous_patterns, safe_alternatives)
FIX_PATTERNS: Dict[str, Dict[str, Any]] = {
    "B102": {
        "dangerous_calls": {"eval", "exec"},
        "safe_hint": "Use ast.literal_eval() for safe evaluation or avoid dynamic code execution",
    },
    "B605": {
        "dangerous_calls": {"os.system", "os.popen"},
        "safe_calls": {"subprocess.run", "subprocess.call"},
        "safe_hint": "Use subprocess.run() with a list of arguments",
    },
    "B602": {
        "check_shell_true": True,
        "safe_hint": "Use shell=False with a list of arguments",
    },
    "B301": {
        "dangerous_calls": {
            "pickle.loads",
            "pickle.load",
            "cPickle.loads",
            "cPickle.load",
        },
        "safe_calls": {"json.loads", "json.load"},
        "safe_hint": "Use json module for data serialization",
    },
    "B608": {
        "check_sql_formatting": True,
        "safe_hint": "Use parameterized queries with ? or %s placeholders",
    },
    "B110": {
        "check_bare_except": True,
        "safe_hint": "Catch specific exceptions like Exception, ValueError, etc.",
    },
    "B105": {
        "check_hardcoded_secrets": True,
        "safe_hint": "Use os.environ.get() or a secrets manager",
    },
    "B106": {
        "check_hardcoded_secrets": True,
        "safe_hint": "Use os.environ.get() or a secrets manager",
    },
    "B506": {
        "dangerous_calls": {"yaml.load", "yaml.unsafe_load"},
        "safe_calls": {"yaml.safe_load"},
        "safe_hint": "Use yaml.safe_load() or pass Loader=yaml.SafeLoader",
    },
    "WEAK_RANDOM": {
        "dangerous_calls": {
            "random.randint",
            "random.random",
            "random.choice",
            "random.randrange",
        },
        "safe_calls": {"secrets.token_hex", "secrets.token_urlsafe", "os.urandom"},
        "safe_hint": "Use secrets module for cryptographic randomness",
    },
    "RESOURCE_LEAK": {
        "check_file_with": True,
        "safe_hint": "Use 'with open(...) as f:' context manager",
    },
    "UNUSED_IMPORT": {
        "check_import_removed": True,
        "safe_hint": "Remove the unused import statement",
    },
}


def verify_fix(
    original_code: str,
    fix_code: str,
    issue_check_id: str,
    issue_line: int,
) -> FixResult:
    """Verify that a submitted fix correctly addresses an issue.

    Checks:
    1. Syntax: fix_code must parse without errors
    2. Pattern removal: the dangerous pattern should be gone
    3. Safe pattern: the correct alternative should be present (where applicable)
    4. No regression: fix should not introduce new dangerous patterns

    Args:
        original_code: The original code snippet
        fix_code: The agent's proposed fix
        issue_check_id: The check_id of the issue being fixed (e.g. "B102")
        issue_line: The line number of the original issue

    Returns:
        FixResult with is_valid, score, feedback, and any new issues introduced
    """
    # Step 1: Syntax check
    try:
        fix_tree = ast.parse(fix_code)
    except SyntaxError as e:
        return FixResult(
            is_valid=False,
            score=0.0,
            feedback=f"Fix has a syntax error at line {e.lineno}: {e.msg}",
        )

    # Step 2: Check that the original dangerous pattern is removed
    pattern_info = FIX_PATTERNS.get(issue_check_id)
    if not pattern_info:
        # Unknown check_id — just verify syntax is OK
        return FixResult(
            is_valid=True,
            score=0.5,
            feedback="Fix parses correctly but could not verify pattern removal "
            f"(unknown check_id: {issue_check_id}).",
        )

    score = 0.0
    feedback_parts: List[str] = []

    # Check dangerous calls removed
    dangerous_still_present = False
    if "dangerous_calls" in pattern_info:
        fix_calls = _collect_call_names(fix_tree)
        for dc in pattern_info["dangerous_calls"]:
            if dc in fix_calls:
                dangerous_still_present = True
                feedback_parts.append(
                    f"Dangerous call '{dc}' is still present in the fix."
                )

    # Check shell=True removed (B602)
    if pattern_info.get("check_shell_true"):
        if _has_shell_true(fix_tree):
            dangerous_still_present = True
            feedback_parts.append("subprocess call still uses shell=True.")

    # Check bare except removed (B110)
    if pattern_info.get("check_bare_except"):
        bare_excepts = _check_bare_except(fix_tree)
        if bare_excepts:
            dangerous_still_present = True
            feedback_parts.append("Bare 'except:' clause is still present.")

    # Check hardcoded secrets removed (B105/B106)
    if pattern_info.get("check_hardcoded_secrets"):
        fix_lines = fix_code.splitlines()
        secret_findings = _check_hardcoded_secrets(fix_tree, fix_lines)
        if secret_findings:
            dangerous_still_present = True
            feedback_parts.append("Hardcoded secret is still present in the fix.")

    # Check SQL formatting removed (B608)
    if pattern_info.get("check_sql_formatting"):
        fix_lines = fix_code.splitlines()
        sql_findings = _check_sql_injection(fix_tree, fix_lines)
        if sql_findings:
            dangerous_still_present = True
            feedback_parts.append("SQL injection vulnerability is still present.")

    # Check file open with 'with' (RESOURCE_LEAK)
    if pattern_info.get("check_file_with"):
        leak_findings = _check_file_no_with(fix_tree)
        if leak_findings:
            dangerous_still_present = True
            feedback_parts.append("File is still opened without 'with' statement.")

    # Check import removed (UNUSED_IMPORT)
    if pattern_info.get("check_import_removed"):
        # Re-check unused imports — the specific one should be gone
        unused_findings = _check_unused_imports(fix_tree)
        if unused_findings:
            # This is acceptable if fewer unused imports remain
            pass  # Don't penalize — we check regression separately

    if not dangerous_still_present:
        score += 0.5
        feedback_parts.append("Dangerous pattern successfully removed.")
    else:
        feedback_parts.append(
            f"Hint: {pattern_info.get('safe_hint', 'Fix the issue.')}"
        )

    # Step 3: Check safe alternative is present
    if "safe_calls" in pattern_info and not dangerous_still_present:
        fix_calls = _collect_call_names(fix_tree)
        has_safe = any(sc in fix_calls for sc in pattern_info["safe_calls"])
        if has_safe:
            score += 0.3
            feedback_parts.append("Safe alternative correctly used.")
        else:
            score += 0.1  # Pattern removed but didn't use the ideal alternative
            feedback_parts.append(
                "Pattern removed but ideal safe alternative not detected."
            )

    elif not dangerous_still_present:
        score += 0.3  # Give credit for removal even without specific safe check

    # Step 4: Regression check — does the fix introduce NEW dangerous patterns?
    original_findings = analyze_code(original_code)
    fix_findings = analyze_code(fix_code)

    original_check_ids = {(f.check_id, f.line_number) for f in original_findings}
    new_issues = []
    for f in fix_findings:
        if (f.check_id, f.line_number) not in original_check_ids:
            # This is a new issue introduced by the fix
            if f.check_id != "UNUSED_IMPORT":  # Don't penalize for import changes
                new_issues.append(
                    f"{f.check_id} at line {f.line_number}: {f.description}"
                )

    if new_issues:
        score = max(0.0, score - 0.3)
        feedback_parts.append(
            f"Fix introduces {len(new_issues)} new issue(s): {'; '.join(new_issues[:2])}"
        )
        return FixResult(
            is_valid=False,
            score=score,
            feedback=" | ".join(feedback_parts),
            issues_introduced=new_issues,
        )

    if not dangerous_still_present:
        score += 0.2  # No regression bonus
        feedback_parts.append("No new issues introduced.")

    return FixResult(
        is_valid=not dangerous_still_present and score >= 0.5,
        score=min(1.0, score),
        feedback=" | ".join(feedback_parts),
    )


# ---------------------------------------------------------------------------
# Fix verification helpers
# ---------------------------------------------------------------------------


def _collect_call_names(tree: ast.AST) -> Set[str]:
    """Collect all function call names in the AST."""
    names: Set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            name = _get_call_name(node)
            if name:
                names.add(name)
    return names


def _has_shell_true(tree: ast.AST) -> bool:
    """Check if any subprocess call has shell=True."""
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func_name = _get_call_name(node)
            if func_name and "subprocess" in func_name:
                for kw in node.keywords:
                    if kw.arg == "shell":
                        if (
                            isinstance(kw.value, ast.Constant)
                            and kw.value.value is True
                        ):
                            return True
    return False
