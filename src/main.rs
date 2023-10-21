mod ir;
mod sexpr;
use sexpr::*;

fn main()  {
  let ctx = &mut ParseContext::new("(do (var a 10) (loop (gt a 0) (set a (- a 1))))");
  let parsed = SExp::parse(ctx).unwrap();

  let mut top_level_func = ir::Func::new();
  let res = top_level_func.comp_expr(&parsed, true);
  if res.is_err() {
    println!("got err {}", res.err().unwrap())
  }

  top_level_func.instructions.iter().for_each(|inst| println!("{}", inst));
}
