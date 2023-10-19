mod sexpr;
mod ir;
use sexpr::*;

fn main() {
    println!("{:?}", SEXP_STRUCT.buffer_encode());
    let ctx = &mut ParseContext::new(SEXP_STRING_IN);
    println!("{:?}", SExp::parse(ctx));
}
