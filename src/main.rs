mod codegen;
mod ir;
mod sexpr;

fn main() {
    let res = ir::IrContext::gen_ir(std::io::stdout(), "(+ 1 2)");
    if res.is_err() {
        println!("Failed compilation: {}", res.err().unwrap())
    }
}
