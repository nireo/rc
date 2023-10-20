mod ir;
mod sexpr;
use sexpr::*;

fn main()  {
  println!("{:?}", SEXP_STRUCT.buffer_encode());
  let ctx = &mut ParseContext::new("(+ 1 2)");
  let parsed = SExp::parse(ctx).unwrap();
}
