; Go tree-sitter queries for structural extraction

; Function declarations
(function_declaration
  name: (identifier) @fn_name) @fn_def

; Method declarations (receiver functions)
(method_declaration
  name: (field_identifier) @fn_name) @fn_def

; Type declarations (structs, interfaces)
(type_declaration
  (type_spec
    name: (type_identifier) @cls_name)) @cls_def

; Import declarations
(import_declaration) @imp_def

; Import spec (individual import path)
(import_spec
  path: (interpreted_string_literal) @imp_source)

; Const declarations
(const_declaration
  (const_spec
    name: (identifier) @const_name))

; Var declarations at package level
(source_file
  (var_declaration
    (var_spec
      name: (identifier) @const_name)))
