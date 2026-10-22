; Bash tree-sitter queries for structural extraction

; Function definitions
(function_definition
  name: (_) @fn_name) @fn_def

; Source/dot commands (imports)
(command
  name: (command_name) @_cmd
  argument: (_) @imp_source
  (#match? @_cmd "^(source|\\.)$"))

; Top-level variable assignments (constants)
(program
  (variable_assignment
    name: (variable_name) @const_name))
