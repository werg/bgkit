; JavaScript tree-sitter queries for structural extraction

; Function declarations
(function_declaration
  name: (identifier) @fn_name) @fn_def

; Arrow functions assigned to variables
(lexical_declaration
  (variable_declarator
    name: (identifier) @fn_name
    value: (arrow_function) @fn_arrow)) @fn_def

; Class declarations
(class_declaration
  name: (identifier) @cls_name) @cls_def

; Method definitions inside classes
(method_definition
  name: (property_identifier) @method_name) @method_def

; Import statements
(import_statement
  source: (string) @imp_source) @imp_def

; Top-level const/let/var declarations (constants)
(program
  (lexical_declaration
    (variable_declarator
      name: (identifier) @const_name)))

; Export statements
(export_statement) @export_def
