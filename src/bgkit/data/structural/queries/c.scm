; C tree-sitter queries for structural extraction

; Function definitions
(function_definition
  declarator: (function_declarator
    declarator: (identifier) @fn_name)) @fn_def

; Pointer function definitions
(function_definition
  declarator: (pointer_declarator
    declarator: (function_declarator
      declarator: (identifier) @fn_name))) @fn_def

; Struct/union type definitions
(type_definition
  type: (struct_specifier
    name: (type_identifier) @cls_name)) @cls_def

(type_definition
  type: (union_specifier
    name: (type_identifier) @cls_name)) @cls_def

; Struct specifiers at top level
(struct_specifier
  name: (type_identifier) @cls_name) @cls_def

; Include directives (imports)
(preproc_include
  path: (_) @imp_source) @imp_def

; Preprocessor defines (constants)
(preproc_def
  name: (identifier) @const_name) @const_def
