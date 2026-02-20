; C++ tree-sitter queries for structural extraction

; Function definitions
(function_definition
  declarator: (function_declarator
    declarator: (identifier) @fn_name)) @fn_def

; Qualified function definitions (namespace::func)
(function_definition
  declarator: (function_declarator
    declarator: (qualified_identifier) @fn_name)) @fn_def

; Class specifiers
(class_specifier
  name: (type_identifier) @cls_name) @cls_def

; Struct specifiers
(struct_specifier
  name: (type_identifier) @cls_name) @cls_def

; Include directives (imports)
(preproc_include
  path: (_) @imp_source) @imp_def

; Namespace definitions
(namespace_definition
  name: (namespace_identifier) @ns_name) @ns_def

; Preprocessor defines (constants)
(preproc_def
  name: (identifier) @const_name) @const_def
