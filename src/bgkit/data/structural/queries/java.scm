; Java tree-sitter queries for structural extraction

; Method declarations
(method_declaration
  name: (identifier) @fn_name) @fn_def

; Constructor declarations
(constructor_declaration
  name: (identifier) @fn_name) @fn_def

; Class declarations
(class_declaration
  name: (identifier) @cls_name) @cls_def

; Interface declarations
(interface_declaration
  name: (identifier) @cls_name) @cls_def

; Import declarations
(import_declaration) @imp_def

; Enum declarations
(enum_declaration
  name: (identifier) @cls_name) @cls_def

; Constant fields (static final)
(field_declaration
  (modifiers) @_mods
  declarator: (variable_declarator
    name: (identifier) @const_name))
