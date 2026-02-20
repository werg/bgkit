; PHP tree-sitter queries for structural extraction

; Function definitions
(function_definition
  name: (name) @fn_name) @fn_def

; Method declarations
(method_declaration
  name: (name) @fn_name) @fn_def

; Class declarations
(class_declaration
  name: (name) @cls_name) @cls_def

; Interface declarations
(interface_declaration
  name: (name) @cls_name) @cls_def

; Trait declarations
(trait_declaration
  name: (name) @cls_name) @cls_def

; Namespace use declarations (imports)
(namespace_use_declaration) @imp_def

; Namespace definitions
(namespace_definition
  name: (namespace_name) @imp_ns)

; Const declarations
(const_declaration
  (const_element
    (name) @const_name))
