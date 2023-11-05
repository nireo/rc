mod codegen;
mod ir;
mod sexpr;

use std::{cell::RefCell, rc::Rc};

use ir::*;
use sexpr::*;

use crate::codegen::{Codegen, ExecContext};

fn main() -> anyhow::Result<()> {
    let input = "(+ 1 2)";

    let input_str = format!("(def (main int) () (do {}))", input);
    let parse_ctx = &mut ParseContext::new(input_str.as_str());
    let parsed_expressions = SExp::parse(parse_ctx).unwrap();

    println!("{:?}", parsed_expressions);

    let top_level_func = Rc::new(RefCell::new(Func::new(None)));
    let mut ir_context = IrContext::new(top_level_func);

    let func = ir_context.scan_func(&parsed_expressions)?;
    ir_context.curr = func;
    ir_context.comp_func(&parsed_expressions)?;

    let mut codegen = Codegen::new(ir_context);
    codegen.codegen_mem()?;

    println!("{:?}", codegen.buffer);
    let exec_ctx = ExecContext::new(&codegen.buffer);
    std::process::exit(exec_ctx.invoke() as i32);
}
