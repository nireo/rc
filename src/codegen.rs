// Codegen handles executing and creating machinecode for a given rc program.

use crate::ir::*;
use std::collections::HashMap;
use std::ffi::c_void;
use std::ptr;

const REG_A: u64 = 0;
const REG_C: u64 = 1;
const REG_D: u64 = 2;
const REG_B: u64 = 3;
const REG_SP: u64 = 4;
const REG_BP: u64 = 5;
const REG_SI: u64 = 6;
const REG_DI: u64 = 7;

// Codegen creates the encoded instructions but doesn't execute the code.
pub struct Codegen<'a> {
  buffer: Vec<u8>, // binary encoded instructions
  jumps: HashMap<i64, Vec<i64>>,
  calls: HashMap<usize, Vec<i64>>,
  fn_to_offset: HashMap<usize, usize>,
  ir_context: IrContext<'a>,
}

impl<'a> Codegen<'a> {
  pub fn new(ir_context: IrContext<'a>) -> Self {
    Self {
      buffer: Vec::new(),
      jumps: HashMap::new(),
      calls: HashMap::new(),
      fn_to_offset: HashMap::new(),
      ir_context,
    }
  }
}

pub struct ExecContext {
  code: *mut c_void,
  stack: *mut c_void,
  code_size: usize,
  stack_size: usize,
  cfunc: extern "C" fn(*mut c_void) -> i64,
}

impl ExecContext {
  pub fn new(code: &[u8]) -> ExecContext {
    let code_size = code.len();
    let stack_size = 8 << 20; // 8MB

    unsafe {
      // Map the code into executable memory
      let code_ptr = libc::mmap(
        ptr::null_mut(),
        code_size,
        libc::PROT_EXEC | libc::PROT_READ | libc::PROT_WRITE,
        libc::MAP_PRIVATE | libc::MAP_ANONYMOUS,
        -1,
        0,
      );
      ptr::copy_nonoverlapping(code.as_ptr(), code_ptr as *mut u8, code_size);

      // Map the stack into memory
      let stack_ptr = libc::mmap(
        ptr::null_mut(),
        stack_size,
        libc::PROT_READ | libc::PROT_WRITE,
        libc::MAP_PRIVATE | libc::MAP_ANONYMOUS,
        -1,
        0,
      );

      // Cast the code pointer to a function pointer
      let cfunc = std::mem::transmute::<*mut c_void, extern "C" fn(*mut c_void) -> i64>(code_ptr);

      ExecContext {
        code: code_ptr,
        stack: stack_ptr,
        code_size,
        stack_size,
        cfunc,
      }
    }
  }

  pub fn invoke(&self) -> i64 {
    (self.cfunc)(self.stack)
  }

  pub fn close(&self) {
    unsafe {
      libc::munmap(self.code, self.code_size);
      libc::munmap(self.stack, self.stack_size);
    }
  }
}

impl Drop for ExecContext {
  fn drop(&mut self) {
    self.close();
  }
}
