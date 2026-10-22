; Python tree-sitter queries for structural extraction

; Functions (top-level and nested)
(function_definition
  name: (identifier) @fn_name) @fn_def

; Classes with optional superclasses
(class_definition
  name: (identifier) @cls_name
  superclasses: (argument_list (identifier) @cls_base)?) @cls_def

; import X / import X.Y
(import_statement
  name: (dotted_name) @imp_module) @imp_def

; from X import Y
(import_from_statement
  module_name: (_) @imp_from_module) @imp_from_def

; Top-level assignments (constants)
(module
  (assignment
    left: (identifier) @const_name))
