mod ir;
mod sexpr;
use sexpr::*;
use std::rc::Rc;
use std::cell::RefCell;

fn main()  {
  let ctx = &mut ParseContext::new("(if (+ 151 215) (+ 120 123))");
  let parsed = SExp::parse(ctx).unwrap();

  let top_level_func = Rc::new(RefCell::new(ir::Func::new(None)));
  let ir_context = ir::IrContext::new(top_level_func);

  let res = ir_context.comp_expr(&parsed, true);
  if res.is_err() {
    println!("got err {}", res.err().unwrap())
  }

  ir_context.curr.borrow().instructions.iter().for_each(|inst| println!("{}", inst));
}
