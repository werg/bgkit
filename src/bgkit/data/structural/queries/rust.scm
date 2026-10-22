; Rust tree-sitter queries for structural extraction

; Function items
(function_item
  name: (identifier) @fn_name) @fn_def

; Struct items
(struct_item
  name: (type_identifier) @cls_name) @cls_def

; Enum items
(enum_item
  name: (type_identifier) @cls_name) @cls_def

; Trait items
(trait_item
  name: (type_identifier) @cls_name) @cls_def

; Impl items
(impl_item) @impl_def

; Use declarations (imports)
(use_declaration
  argument: (_) @imp_path) @imp_def

; Const items
(const_item
  name: (identifier) @const_name)

; Static items
(static_item
  name: (identifier) @const_name)
