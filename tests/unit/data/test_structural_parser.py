"""Tests for the tree-sitter structural extraction parser."""

from __future__ import annotations

from bgkit.data.structural.parser import (
    LANGUAGE_MAP,
    TIER_A_LANGUAGES,
    FileSkeleton,
    parse_file,
    parse_repo,
)

# ---------------------------------------------------------------------------
# Python parsing
# ---------------------------------------------------------------------------

PYTHON_SNIPPET = """\
import os
from pathlib import Path
from . import utils
from ..base import BaseClass

CONST = 42
MAX_SIZE = 100
_private = True

class Foo(BaseClass):
    def method(self, x: int) -> str:
        return str(x)

class Bar(Foo):
    pass

def standalone(a, b):
    return a + b
"""


def test_parse_python_functions():
    skeleton = parse_file("example.py", PYTHON_SNIPPET, "Python")
    assert skeleton is not None
    assert skeleton.tier == "A"
    assert skeleton.language == "Python"

    # Top-level functions only (methods are on classes)
    fn_names = [f.name for f in skeleton.functions]
    assert "standalone" in fn_names

    # Methods should NOT be in top-level functions
    assert "method" not in fn_names


def test_parse_python_classes():
    skeleton = parse_file("example.py", PYTHON_SNIPPET, "Python")
    assert skeleton is not None

    cls_names = [c.name for c in skeleton.classes]
    assert "Foo" in cls_names
    assert "Bar" in cls_names

    foo_cls = next(c for c in skeleton.classes if c.name == "Foo")
    assert "BaseClass" in foo_cls.bases

    # Check methods are attached to classes
    method_names = [m.name for m in foo_cls.methods]
    assert "method" in method_names


def test_parse_python_imports():
    skeleton = parse_file("example.py", PYTHON_SNIPPET, "Python")
    assert skeleton is not None

    import_modules = [i.module for i in skeleton.imports]
    assert "os" in import_modules

    # from pathlib import Path
    pathlib_imp = next((i for i in skeleton.imports if i.module == "pathlib"), None)
    assert pathlib_imp is not None
    assert "Path" in pathlib_imp.names

    # Relative imports
    relative_imports = [i for i in skeleton.imports if i.is_relative]
    assert len(relative_imports) >= 2


def test_parse_python_constants():
    skeleton = parse_file("example.py", PYTHON_SNIPPET, "Python")
    assert skeleton is not None

    # Only UPPER_CASE names should be captured as constants
    assert "CONST" in skeleton.constants
    assert "MAX_SIZE" in skeleton.constants
    # _private is underscore-prefixed and lowercase, should not be a constant
    assert "_private" not in skeleton.constants


# ---------------------------------------------------------------------------
# JavaScript parsing
# ---------------------------------------------------------------------------

JS_SNIPPET = """\
import { foo, bar } from 'module-a';
import baz from './local';

const MAX_SIZE = 100;

class Animal extends Base {
    constructor(name) {
        this.name = name;
    }
    speak() {
        return this.name;
    }
}

function greet(name) {
    return `Hello ${name}`;
}
"""


def test_parse_javascript_functions():
    skeleton = parse_file("example.js", JS_SNIPPET, "JavaScript")
    assert skeleton is not None
    assert skeleton.tier == "A"

    fn_names = [f.name for f in skeleton.functions]
    assert "greet" in fn_names


def test_parse_javascript_classes():
    skeleton = parse_file("example.js", JS_SNIPPET, "JavaScript")
    assert skeleton is not None

    cls_names = [c.name for c in skeleton.classes]
    assert "Animal" in cls_names

    animal = next(c for c in skeleton.classes if c.name == "Animal")
    assert "Base" in animal.bases


def test_parse_javascript_imports():
    skeleton = parse_file("example.js", JS_SNIPPET, "JavaScript")
    assert skeleton is not None

    import_modules = [i.module for i in skeleton.imports]
    assert "module-a" in import_modules
    assert "./local" in import_modules

    local_imp = next(i for i in skeleton.imports if i.module == "./local")
    assert local_imp.is_relative


def test_parse_javascript_constants():
    skeleton = parse_file("example.js", JS_SNIPPET, "JavaScript")
    assert skeleton is not None

    assert "MAX_SIZE" in skeleton.constants


# ---------------------------------------------------------------------------
# Go parsing
# ---------------------------------------------------------------------------

GO_SNIPPET = """\
package main

import (
    "fmt"
    "os"
)

const MaxSize = 100

type Animal struct {
    Name string
}

func (a *Animal) Speak() string {
    return a.Name
}

func greet(name string) string {
    return fmt.Sprintf("Hello %s", name)
}
"""


def test_parse_go_functions():
    skeleton = parse_file("main.go", GO_SNIPPET, "Go")
    assert skeleton is not None
    assert skeleton.tier == "A"

    fn_names = [f.name for f in skeleton.functions]
    assert "greet" in fn_names

    # Speak should be a method on Animal
    animal = next((c for c in skeleton.classes if c.name == "Animal"), None)
    assert animal is not None
    method_names = [m.name for m in animal.methods]
    assert "Speak" in method_names


def test_parse_go_imports():
    skeleton = parse_file("main.go", GO_SNIPPET, "Go")
    assert skeleton is not None

    import_modules = [i.module for i in skeleton.imports]
    assert "fmt" in import_modules
    assert "os" in import_modules


def test_parse_go_types():
    skeleton = parse_file("main.go", GO_SNIPPET, "Go")
    assert skeleton is not None

    cls_names = [c.name for c in skeleton.classes]
    assert "Animal" in cls_names


def test_parse_go_constants():
    skeleton = parse_file("main.go", GO_SNIPPET, "Go")
    assert skeleton is not None

    assert "MaxSize" in skeleton.constants


# ---------------------------------------------------------------------------
# Tier A vs Tier B
# ---------------------------------------------------------------------------

def test_tier_a_language_uses_custom_queries():
    """Tier A languages should use .scm queries and report tier='A'."""
    skeleton = parse_file("test.py", "def foo(): pass\n", "Python")
    assert skeleton is not None
    assert skeleton.tier == "A"


def test_tier_b_language_uses_generic_extraction():
    """Tier B languages should fall back to generic extraction and report tier='B'."""
    # Kotlin is in LANGUAGE_MAP but not in TIER_A_LANGUAGES
    kt_code = """\
fun greet(name: String): String {
    return "Hello $name"
}

class Animal(val name: String) {
    fun speak(): String = name
}
"""
    skeleton = parse_file("example.kt", kt_code, "Kotlin")
    assert skeleton is not None
    assert skeleton.tier == "B"
    # Should still extract some functions via generic node types
    assert len(skeleton.functions) > 0 or len(skeleton.classes) > 0


# ---------------------------------------------------------------------------
# Unsupported / edge cases
# ---------------------------------------------------------------------------

def test_unsupported_language_returns_none():
    """A language not in LANGUAGE_MAP should return None."""
    result = parse_file("test.xyz", "some content", "Brainfuck")
    assert result is None


def test_none_language_returns_none():
    """None language should return None."""
    result = parse_file("test.txt", "hello", None)
    assert result is None


def test_empty_file():
    """An empty file should parse without error."""
    skeleton = parse_file("empty.py", "", "Python")
    assert skeleton is not None
    assert skeleton.functions == []
    assert skeleton.classes == []
    assert skeleton.imports == []


def test_language_map_covers_tier_a():
    """All Tier A tree-sitter names should be reachable from LANGUAGE_MAP."""
    ts_names_in_map = set(LANGUAGE_MAP.values())
    for ts_name in TIER_A_LANGUAGES:
        assert ts_name in ts_names_in_map, f"Tier A language '{ts_name}' not in LANGUAGE_MAP"


# ---------------------------------------------------------------------------
# parse_repo
# ---------------------------------------------------------------------------

def test_parse_repo_with_mock_snapshot():
    """parse_repo should iterate over files and collect skeletons."""
    from bgkit.data.repo_processing import FileRecord, RepoSnapshot

    snapshot = RepoSnapshot(
        repo_path="/tmp/test-repo",
        commit_sha="abc123",
        files=[
            FileRecord(
                path="main.py",
                content="def hello(): pass\n",
                size_bytes=18,
                language="Python",
            ),
            FileRecord(
                path="app.js",
                content="function greet() { return 'hi'; }\n",
                size_bytes=35,
                language="JavaScript",
            ),
            FileRecord(
                path="README.md",
                content="# Hello\n",
                size_bytes=8,
                language="Markdown",
            ),
        ],
    )

    skeletons = parse_repo(snapshot)

    # Python and JS should parse; Markdown may or may not depending on tree-sitter
    py_skeleton = next((s for s in skeletons if s.path == "main.py"), None)
    assert py_skeleton is not None
    assert py_skeleton.tier == "A"
    fn_names = [f.name for f in py_skeleton.functions]
    assert "hello" in fn_names

    js_skeleton = next((s for s in skeletons if s.path == "app.js"), None)
    assert js_skeleton is not None
    fn_names = [f.name for f in js_skeleton.functions]
    assert "greet" in fn_names


# ---------------------------------------------------------------------------
# Rust parsing
# ---------------------------------------------------------------------------

RUST_SNIPPET = """\
use std::collections::HashMap;
use crate::utils;

const MAX: usize = 100;

struct Animal {
    name: String,
}

impl Animal {
    fn new(name: String) -> Self {
        Animal { name }
    }
    fn speak(&self) -> &str {
        &self.name
    }
}

fn greet(name: &str) -> String {
    format!("Hello {}", name)
}
"""


def test_parse_rust_functions():
    skeleton = parse_file("lib.rs", RUST_SNIPPET, "Rust")
    assert skeleton is not None
    assert skeleton.tier == "A"

    fn_names = [f.name for f in skeleton.functions]
    assert "greet" in fn_names
    # new and speak should be methods on Animal
    assert "new" not in fn_names
    assert "speak" not in fn_names


def test_parse_rust_structs():
    skeleton = parse_file("lib.rs", RUST_SNIPPET, "Rust")
    assert skeleton is not None

    cls_names = [c.name for c in skeleton.classes]
    assert "Animal" in cls_names

    animal = next(c for c in skeleton.classes if c.name == "Animal")
    method_names = [m.name for m in animal.methods]
    assert "new" in method_names
    assert "speak" in method_names


def test_parse_rust_imports():
    skeleton = parse_file("lib.rs", RUST_SNIPPET, "Rust")
    assert skeleton is not None

    imp_modules = [i.module for i in skeleton.imports]
    assert any("std" in m for m in imp_modules)


def test_parse_rust_constants():
    skeleton = parse_file("lib.rs", RUST_SNIPPET, "Rust")
    assert skeleton is not None

    assert "MAX" in skeleton.constants


# ---------------------------------------------------------------------------
# Java parsing
# ---------------------------------------------------------------------------

JAVA_SNIPPET = """\
package com.example;

import java.util.List;

public class Animal extends Base {
    public static final int MAX = 100;

    public Animal(String name) {
        this.name = name;
    }

    public String speak() {
        return this.name;
    }
}
"""


def test_parse_java_classes():
    skeleton = parse_file("Animal.java", JAVA_SNIPPET, "Java")
    assert skeleton is not None
    assert skeleton.tier == "A"

    cls_names = [c.name for c in skeleton.classes]
    assert "Animal" in cls_names

    animal = next(c for c in skeleton.classes if c.name == "Animal")
    assert "Base" in animal.bases

    method_names = [m.name for m in animal.methods]
    assert "Animal" in method_names  # Constructor
    assert "speak" in method_names


def test_parse_java_imports():
    skeleton = parse_file("Animal.java", JAVA_SNIPPET, "Java")
    assert skeleton is not None

    imp_modules = [i.module for i in skeleton.imports]
    assert "java.util.List" in imp_modules
