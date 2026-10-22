; Ruby tree-sitter queries for structural extraction

; Method definitions
(method
  name: (_) @fn_name) @fn_def

; Singleton method definitions (self.method)
(singleton_method
  name: (_) @fn_name) @fn_def

; Class definitions
(class
  name: (_) @cls_name) @cls_def

; Module definitions
(module
  name: (_) @cls_name) @cls_def

; Require calls (imports) — require 'foo'
(call
  method: (identifier) @_method_name
  arguments: (argument_list (string) @imp_source)
  (#match? @_method_name "^require"))

; Top-level constant assignments
(program
  (assignment
    left: (constant) @const_name))
