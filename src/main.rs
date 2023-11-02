mod codegen;
mod ir;
mod sexpr;

fn main() {
    let res = ir::IrContext::gen_ir(std::io::stdout(), "(+ 3 6)");
    if res.is_err() {
        println!("Failed compilation: {}", res.err().unwrap())
    }
}
