// Codegen handles executing and creating machinecode for a given rc program.
extern crate libc;

use crate::ir::*;
use anyhow::anyhow;
use anyhow::Result;
use byteorder::{LittleEndian, WriteBytesExt};
use std::cell::RefCell;
use std::collections::HashMap;
use std::ffi::c_void;
use std::ptr;
use std::rc::Rc;
use thiserror::Error;

#[derive(Error, Debug)]
enum CodegenErrors {
    #[error("unknown binop operation")]
    UnknownBinop,
}

const REG_A: u8 = 0;
const REG_C: u8 = 1;
const REG_D: u8 = 2;
const REG_B: u8 = 3;
const REG_SP: u8 = 4;
const REG_BP: u8 = 5;
const REG_SI: u8 = 6;
const REG_DI: u8 = 7;

// Codegen creates the encoded instructions but doesn't execute the code.
pub struct Codegen<'a> {
    pub buffer: Vec<u8>, // binary encoded instructions
    jumps: HashMap<i64, Vec<usize>>,
    calls: HashMap<i64, Vec<usize>>,
    fn_to_offset: Vec<usize>,
    ir_context: IrContext<'a>,
    alignment: usize,
}

impl<'a> Codegen<'a> {
    pub fn new(ir_context: IrContext<'a>) -> Self {
        Self {
            buffer: Vec::new(),
            jumps: HashMap::new(),
            calls: HashMap::new(),
            fn_to_offset: vec![0],
            ir_context,
            alignment: 16,
        }
    }

    fn mem_entry(&mut self) -> Result<()> {
        self.buffer.extend_from_slice(&[0x53, 0x48, 0x89, 0xFB]);
        self.asm_call(0)?;
        self.buffer.extend_from_slice(&[0x48, 0x8b, 0x03]); // mov rax, [rbx]
        self.buffer.push(0x5b); // pop rbx
        self.buffer.push(0xc3); // ret
        Ok(())
    }

    fn mem_end(&mut self) {
        for (label, offset_list) in &self.calls {
            let dst_offset = self.fn_to_offset[*label as usize] as i32;
            for patch_off in offset_list {
                let src_offset = *patch_off as i32 + 4;
                let rel = ((dst_offset - src_offset) as i32).to_le_bytes();
                self.buffer[*patch_off..*patch_off + 4].copy_from_slice(&rel);
            }
        }
        self.calls.clear();
        self.padding();
    }

    fn padding(&mut self) {
        if self.alignment == 0 {
            return;
        } else {
            self.buffer.push(0xcc);
            while self.buffer.len() % self.alignment != 0 {
                self.buffer.push(0xcc);
            }
        }
    }

    pub fn codegen_mem(&mut self) -> Result<()> {
        self.mem_entry()?;
        for func in self.ir_context.funcs[1..].to_vec() {
            self.func(func)?;
        }
        self.mem_end();
        Ok(())
    }

    fn func(&mut self, func_to_compile: Rc<RefCell<Func<'a>>>) -> Result<()> {
        self.padding();
        self.fn_to_offset.push(self.buffer.len()); // function index -> code offset

        let fn_borrow = func_to_compile.borrow();

        // virtual instruction -> code offset
        let mut pos_to_offset = Vec::new();
        for inst in &fn_borrow.instructions {
            pos_to_offset.push(self.buffer.len());
            // handle codegen for each instruction.
            match inst {
                Instruction::Constant(val, dst) => self.constant(*val, *dst)?,
                Instruction::Ret(a) => self.ret(*a)?,
                Instruction::Binop(op, a1, a2, dst, _) => self.binop(op.clone(), *a1, *a2, *dst)?,
                Instruction::Mov(src, dst) => self.mov(*src, *dst)?,
                Instruction::Jmp(label) => self.jmp(*label),
                Instruction::Jmpf(a1, label) => self.jmpf(*a1, *label)?,
                Instruction::Call(index, arg_start, level_curr, level_new) => {
                    self.call(*index, *arg_start, *level_curr, *level_new)?
                }
                Instruction::Unop(op, a1, dst) => self.unop(op.clone(), *a1, *dst)?,
            }
        }

        // fill in jmp address
        for (label, offset_list) in &self.jumps {
            let dst_offset = pos_to_offset[fn_borrow.labels[*label as usize].unwrap() as usize];
            for patch_off in offset_list {
                let src_offset = patch_off + 4;
                let rel = ((dst_offset - src_offset) as i32).to_le_bytes();
                println!("{:?}", rel);
                self.buffer[*patch_off..*patch_off + 4].copy_from_slice(&rel);
            }
        }
        self.jumps.clear();
        Ok(())
    }

    fn ret(&mut self, a1: i64) -> Result<()> {
        if a1 > 0 {
            self.load_rax(a1)?;
            self.store_rax(0)?;
        }
        self.buffer.push(0xc3);
        Ok(())
    }

    fn unop(&mut self, operation: Unop, a1: i64, dst: i64) -> Result<()> {
        self.load_rax(a1)?;
        match operation {
            Unop::Not => {
                self.buffer.extend_from_slice(&[
                    0x48, 0x85, 0xc0, // test rax, rax
                    0x0f, 0x94, 0xc0, // sete al
                    0x0f, 0xb6, 0xc0, // movzx eax, al
                ]);
            }
            Unop::Minus => {
                self.buffer.extend_from_slice(&[0x48, 0xf7, 0xd8]); // neg rax
            }
        }

        self.store_rax(dst)
    }

    fn call(&mut self, index: i64, arg_start: i64, level_curr: i64, level_new: i64) -> Result<()> {
        // put a list of pointers to outer frames in the rsp stack
        if level_new > level_curr {
            self.buffer.push(0x53); // push rbx
        }

        for _ in 0..(std::cmp::min(level_new, level_curr) - 1) {
            // copy previous list
            self.buffer.extend_from_slice(&[0xff, 0xb4, 0x24]);
            self.buffer
                .write_i32::<LittleEndian>((level_new as i32 - 1) * 8)?;
        }

        // make new frame and call the targer
        if arg_start != 0 {
            self.buffer.extend_from_slice(&[0x48, 0x81, 0xc3]); // add rbx, arg_start * 8
            self.buffer
                .write_i32::<LittleEndian>(arg_start as i32 * 8)?;
        }
        self.asm_call(index - 1)?;
        if arg_start != 0 {
            self.buffer.extend_from_slice(&[0x48, 0x81, 0xc3]); // add rbx, -arg_start * 8
            self.buffer
                .write_i32::<LittleEndian>(-arg_start as i32 * 8)?;
        }

        self.buffer.extend_from_slice(&[0x48, 0x81, 0xc4]);
        self.buffer
            .write_i32::<LittleEndian>((level_new as i32 - 1) * 8)?;
        Ok(())
    }

    fn constant(&mut self, val: i64, dst: i64) -> Result<()> {
        if val == 0 {
            self.buffer.extend_from_slice(&[0x31, 0xc0]);
        } else if val == -1 {
            self.buffer.extend_from_slice(&[0x48, 0x83, 0xc8, 0xff]);
        } else if (val >> 31) == 0 {
            self.buffer.push(0xb8);
            self.buffer.write_i32::<LittleEndian>(val as i32)?;
        } else if (val >> 31) == -1 {
            self.buffer.extend_from_slice(&[0x48, 0x83, 0xc7, 0xc0]);
            self.buffer.write_i32::<LittleEndian>(val as i32)?;
        } else {
            self.buffer.extend_from_slice(&[0x48, 0xb8]);
            self.buffer.write_i64::<LittleEndian>(val)?;
        }
        self.store_rax(dst)
    }

    fn binop(&mut self, op: Binop, a1: i64, a2: i64, dst: i64) -> Result<()> {
        self.load_rax(a1)?;

        let arith: HashMap<Binop, Vec<u8>> = [
            (Binop::Plus, vec![0x48, 0x03]),
            (Binop::Minus, vec![0x48, 0x2b]),
            (Binop::Multiply, vec![0x48, 0x0f, 0xaf]),
        ]
        .iter()
        .cloned()
        .collect();

        let cmp: HashMap<Binop, u8> = [
            (Binop::EQ, 0x94),
            (Binop::NE, 0x95),
            (Binop::GE, 0x9d),
            (Binop::GT, 0x9f),
            (Binop::LE, 0x9e),
            (Binop::LT, 0x9c),
        ]
        .iter()
        .cloned()
        .collect();

        match op {
            Binop::Divide | Binop::Modulo => {
                // xor edx, dex | idiv rax, [rbx + a2 * 8]
                self.buffer
                    .extend_from_slice(&[0x31, 0xd2, 0x48, 0xf7, 0xbb]);
                self.buffer.write_i32::<LittleEndian>(a2 as i32 * 8)?;
                if op == Binop::Modulo {
                    self.buffer.extend_from_slice(&[0x48, 0x89, 0xd0]);
                }
            }
            Binop::And => {
                self.buffer
                    .extend_from_slice(&[0x48, 0x85, 0xc0, 0x0f, 0x95, 0xc0]);
                self.asm_load(REG_D, REG_B, a2 * 8)?;
                self.buffer.extend_from_slice(&[
                    0x48, 0x85, 0xd2, // test rdx, rdx
                    0x0f, 0x95, 0xc2, // setne dl
                    0x21, 0xd0, // and eax, edx
                    0x0f, 0xb6, 0xc0, // movzx eax, al
                ]);
            }
            Binop::Or => {
                self.asm_disp(vec![0x48, 0x0b], REG_A, REG_B, a2 * 8)?;
                self.buffer.extend_from_slice(&[
                    0x0f, 0x95, 0xc0, // setne el
                    0x0f, 0xb6, 0xc0, // movzx eax, al
                ]);
            }
            _ if arith.contains_key(&op) => {
                // op rax, [rbx + a2*8]
                self.asm_disp(arith[&op].clone(), REG_A, REG_B, a2 * 8)?
            }
            _ if cmp.contains_key(&op) => {
                // cmp rax, [rbx + a2*8]
                self.asm_disp(vec![0x48, 0x3b], REG_A, REG_B, a2 * 8)?;
                // setcc al | movzx eax, al
                self.buffer
                    .extend_from_slice(&[0x0f, cmp[&op].clone(), 0xc0, 0x0f, 0xb6, 0xc0]);
            }
            _ => return Err(anyhow!(CodegenErrors::UnknownBinop)),
        }

        self.store_rax(dst)
    }

    fn jmpf(&mut self, a1: i64, label: i64) -> Result<()> {
        self.load_rax(a1)?;
        self.buffer.extend_from_slice(&[
            0x48, 0x85, 0xc0, // test rax, rax
            0x0f, 0x84, // je
        ]);
        self.jumps
            .entry(label)
            .or_insert_with(Vec::new)
            .push(self.buffer.len());
        self.buffer.extend_from_slice(&[0x00, 0x00, 0x00, 0x00]);
        Ok(())
    }

    fn jmp(&mut self, label: i64) {
        self.buffer.push(0xe9); // jmp
        self.jumps
            .entry(label)
            .or_insert_with(Vec::new)
            .push(self.buffer.len());
        self.buffer.extend_from_slice(&[0x00, 0x00, 0x00, 0x00]);
    }

    fn asm_disp(&mut self, mut lead: Vec<u8>, mut reg: u8, mut rm: u8, disp: i64) -> Result<()> {
        assert!(reg < 16 && rm < 16 && rm != REG_SP);
        if reg >= 8 || rm >= 8 {
            assert!((lead[0] >> 4) == 0b0100);
            lead[0] |= (reg >> 3) << 2;
            lead[0] |= (rm >> 3) << 0;
            reg &= 0b111;
            rm &= 0b111;
        }

        self.buffer.extend_from_slice(&lead);
        let md = if disp == 0 {
            0 // [rm]
        } else if -128 <= disp && disp < 128 {
            1 // [rm + disp8]
        } else {
            2 // rm + disp32
        };

        self.buffer.push((md << 6) | (reg << 3) | rm);
        if md == 1 {
            self.buffer.write_i8(disp as i8)?;
        } else if md == 2 {
            self.buffer.write_i32::<LittleEndian>(disp as i32)?;
        }

        Ok(())
    }

    // mov reg, [rm + disp]
    fn asm_load(&mut self, reg: u8, rm: u8, disp: i64) -> Result<()> {
        self.asm_disp(vec![0x48, 0x8b], reg, rm, disp)
    }

    fn asm_store(&mut self, rm: u8, disp: i64, reg: u8) -> Result<()> {
        self.asm_disp(vec![0x48, 0x89], reg, rm, disp)
    }

    fn store_rax(&mut self, dst: i64) -> Result<()> {
        self.asm_store(REG_B, dst * 8, REG_A)
    }

    fn load_rax(&mut self, src: i64) -> Result<()> {
        self.asm_load(REG_A, REG_B, src * 8)
    }

    fn mov(&mut self, src: i64, dst: i64) -> Result<()> {
        if src == dst {
            // already in the same place do nothing
            Ok(())
        } else {
            self.load_rax(src)?;
            self.load_rax(dst)?;
            Ok(())
        }
    }

    fn asm_call(&mut self, l: i64) -> Result<()> {
        self.buffer.push(0xe8);
        self.calls
            .entry(l)
            .or_insert_with(Vec::new)
            .push(self.buffer.len());
        self.buffer.extend_from_slice(&[0x00, 0x00, 0x00, 0x00]);
        Ok(())
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
            let cfunc =
                std::mem::transmute::<*mut c_void, extern "C" fn(*mut c_void) -> i64>(code_ptr);

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
