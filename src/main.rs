mod codegen;
mod ir;
mod sexpr;

use std::{cell::RefCell, rc::Rc};

use ir::*;
use sexpr::*;

use crate::codegen::{Codegen, ExecContext};
use clap::Parser;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    // Should code be generated and executed.
    #[arg(short, long, default_value_t = false)]
    gen: bool,

    #[arg(long, default_value_t = false)]
    dump_ir: bool,

    #[arg(short, long)]
    file_path: String,
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();

    let input = std::fs::read_to_string(args.file_path)?;

    let input_str = format!("(def (main int) () (do {}))", input);
    let parse_ctx = &mut ParseContext::new(input_str.as_str());
    let parsed_expressions = SExp::parse(parse_ctx).unwrap();

    let top_level_func = Rc::new(RefCell::new(Func::new(None)));
    let mut ir_context = IrContext::new(top_level_func);

    let func = ir_context.scan_func(&parsed_expressions)?;
    ir_context.curr = func;
    ir_context.comp_func(&parsed_expressions)?;

    if args.dump_ir {
        ir_context.dump_instructions(std::io::stdout())?;
    }

    if args.gen {
        let mut codegen = Codegen::new(ir_context);
        codegen.codegen_mem()?;

        let exec_ctx = ExecContext::new(&codegen.buffer);
        std::process::exit(exec_ctx.invoke() as i32);
    }

    Ok(())
}
