# rc: a lisp-like language compiler

rc is a compiler for a sexpression based language written using rust.

```
Usage: rc [OPTIONS] --file-path <FILE_PATH>

Options:
  -g, --gen
      --dump-ir
  -f, --file-path <FILE_PATH>
  -h, --help                   Print help
  -V, --version                Print version

```

## Current features

Currently only bare minimum features are implementedÖ

* Binary and unary operations
* Functions calls
* Loops
* Conditional statements
* Custom functions

The compiler works by first emitting a platform-agnostic ir output then compiling that ir output to a given platform. In this case the back-end is currently encodes instructions using hex codes to x64 platform. Might consider looking into a LLVM back-end for better performance. 

## Future features

* Nested functions
* Pointers
* Arrays
* Strings
* Better performance?!
