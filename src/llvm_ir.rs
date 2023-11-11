use inkwell::{
    context::Context,
    module::Module,
    passes::PassManager,
    values::{FunctionValue, PointerValue},
};

pub struct CodegenLLVM<'a, 'ctx> {
    context: &'ctx Context,
    builder: &'a Builder<'ctx>,
    pass_manager: &'a PassManager<FunctionValue<'ctx>>,
    module: &'a Module<'ctx>,
    function: &'a FunctionValue,
    fn_value_opt: Option<FunctionValue<'ctx>>,
}

impl<'a, 'ctx> CodegenLLVM<'a, 'ctx> {
    #[inline]
    fn get_func(&self, name: &str) -> Option<FunctionValue<'ctx>> {
        self.module.get_function(name)
    }

    #[inline]
    fn fn_value(&self) -> FunctionValue<'ctx> {
        self.fn_value_opt.unwrap()
    }

    fn create_entry_block_alloca(&self, name: &str) -> PointerValue<'ctx> {
        let builder = self.context.create_builder();
        let entry = self.fn_value().get_first_basic_block().unwrap();

        match entry.get_first_instruction() {
            Some(first_inst) => builder.position_before(&first_inst),
            None => builder.position_at_end(entry),
        }

        builder.build_alloca(self.context.f64_type(), name).unwrap()
    }
}
