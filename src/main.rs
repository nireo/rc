mod ir;
mod sexpr;
use sexpr::*;

fn main()  {
  let ctx = &mut ParseContext::new("(+ 1 2)");
  let parsed = SExp::parse(ctx).unwrap();

  let mut top_level_func = ir::Func::new();
  let _ = top_level_func.comp_expr(&parsed, true);

  top_level_func.instructions.iter().for_each(|inst| println!("{}", inst));
}
